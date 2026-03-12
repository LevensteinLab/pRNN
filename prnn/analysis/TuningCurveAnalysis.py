from prnn.utils.agent import RandomActionAgent
import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter, label
from scipy.signal import correlate2d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from prnn.utils.general import saveFig


class TuningCurveAnalysis:
    def __init__(
        self,
        predictiveNet,
        timesteps_wake=5000,
        theta="expand",
        start_pos=1,
        EV_thresh=0.5,
    ):
        self.start_pos = start_pos  # the numbering of occupiable locations starts from this
        self.metrics = {}

        env = predictiveNet.EnvLibrary[0]
        action_probability = np.array([0.15, 0.15, 0.6, 0.1, 0, 0, 0])
        agent = RandomActionAgent(env.action_space, action_probability)

        # Calculate Tuning Curves, Spatial Info
        self.tuning_curves, SI, decoder = predictiveNet.calculateSpatialRepresentation(
            env,
            agent,
            timesteps=15000,
            trainDecoder=False,
            trainHDDecoder=False,
            saveTrainingData=True,
            HDinfo=True,
        )
        print("Calculating EV_s")
        WAKEactivity = self.runWAKE(predictiveNet, env, agent, timesteps_wake, theta=theta)
        FAKEactivity, self.metrics["EVs"] = self.calculateTuningCurveReliability(
            WAKEactivity, self.tuning_curves
        )
        self.metrics["SI"] = SI["SI"].values
        self.metrics["HD_info"] = SI["HDinfo"].values

        # Calculate Border Scores
        wallgroups, not_near_walls = LRoomWallGroups(env)
        print("Warning: border score only works with L room currently.")
        border_scores = [
            calculateBorderScore(tc, wallgroups, not_near_walls)
            for tc in self.tuning_curves.values()
        ]
        self.metrics["border_score"] = np.array(border_scores)

        # Calculate autocorrelation-based metrics: peaks, field size, field asymmetry
        self.tc_autocorrs = pf_autocorr(self.tuning_curves, peakNorm=True)
        autocorr_peaks = np.array([count_autocorr_peaks(ac) for ac in self.tc_autocorrs.values()])
        self.metrics["pf_peaks"] = (autocorr_peaks + 1) // 2
        self.metrics["fieldsize"] = np.array(
            [calculate_field_size(ac) for ac in self.tc_autocorrs.values()]
        )
        self.metrics["fieldasymmetry"] = np.array(
            [calculate_field_asymmetry(ac) for ac in self.tc_autocorrs.values()]
        )

        # Determine cell type groups based on calcualted metrics
        self.groupCells(EV_thresh=EV_thresh)

        self.groupFrac, _ = np.histogram(
            self.groupID, bins=np.arange(-0.5, max(self.groupID) + 1.5), density=True
        )
        self.groupFrac_welltuned, _ = np.histogram(
            self.groupID[self.metrics["EVs"] > EV_thresh],
            bins=np.arange(-0.5, max(self.groupID) + 1.5),
            density=True,
        )

        # Fit PCA
        self.pca, self.pca_scaler = fit_pca(self.metrics)

    def groupCells(
        self,
        SI_thresh=0.5,
        EV_unthresh=0.15,
        HD_thresh=0.5,
        border_thresh=0,
        border_symmetrythresh=3,
        EV_thresh=0.5,
        place_symmetrythresh=3,
    ):
        dead = (np.isnan(self.metrics["fieldasymmetry"])) & (np.isnan(self.metrics["fieldsize"]))
        untuned = (
            ~dead
            & (self.metrics["EVs"] <= EV_unthresh)
            & (self.metrics["SI"] <= SI_thresh)
            & (self.metrics["HD_info"] <= HD_thresh)
        )
        HD_cells = (
            ~dead
            & (self.metrics["EVs"] <= EV_unthresh)
            & (self.metrics["SI"] <= SI_thresh)
            & (self.metrics["HD_info"] > HD_thresh)
        )

        border_cells = (
            ~untuned
            & ~HD_cells
            & ~dead
            & (self.metrics["border_score"] > border_thresh)
            & (self.metrics["fieldasymmetry"] > border_symmetrythresh)
        )
        single_field = (
            ~untuned
            & ~HD_cells
            & ~border_cells
            & ~dead
            & (self.metrics["pf_peaks"] == 1)
            & (self.metrics["EVs"] > EV_thresh)
            & (self.metrics["fieldasymmetry"] < place_symmetrythresh)
        )  # & \
        # (self.metrics['SI']>SI_thresh)

        spatial_HD = (
            ~untuned
            & ~HD_cells
            & ~border_cells
            & ~single_field
            & ~dead
            & (self.metrics["SI"] > SI_thresh)
            & (self.metrics["HD_info"] > HD_thresh)
        )

        # multi_field_reliable = ~border_cells & ~untunedCells & ~single_field & ~place_HD & \
        #     (TCA.metrics['pf_peaks']>1) & (TCA.metrics['EVs']>0.5)
        # multi_field_unreliable = ~border_cells & ~untunedCells & ~single_field & ~place_HD & \
        #     (TCA.metrics['pf_peaks']>1) & (TCA.metrics['EVs']<=0.5)
        # single_field_unreliable = ~border_cells & ~untunedCells & ~single_field & ~place_HD & \
        #     ~multi_field_reliable & ~multi_field_unreliable
        complex_cells = ~border_cells & ~single_field & ~untuned & ~HD_cells & ~spatial_HD & ~dead

        groups = {
            "untuned": untuned,
            "HD_cells": HD_cells,
            "single_field": single_field,
            "border_cells": border_cells,
            "spatial_HD": spatial_HD,
            "complex_cells": complex_cells,
            "dead": dead,
        }
        groupID = np.argmax(np.column_stack(list(groups.values())), axis=1)

        self.cellgroups, self.groupID = groups, groupID
        return

    def runWAKE(self, pN, env, agent, timesteps_wake, theta="mean"):
        print("Running WAKE")
        a = {}
        a["obs"], a["act"], a["state"], _ = pN.collectObservationSequence(
            env, agent, timesteps_wake, discretize=True
        )
        a["obs_pred"], a["obs_next"], h = pN.predict(a["obs"], a["act"])

        if theta == "mean":
            h = h.mean(axis=0, keepdims=True)
        if theta == "expand":
            k = h.size(dim=0)
            h = h.transpose(0, 1).reshape((-1, 1, h.size(dim=2))).swapaxes(0, 1)
            a["state"]["agent_pos"] = np.repeat(a["state"]["agent_pos"], k, axis=0)
            a["state"]["agent_pos"] = a["state"]["agent_pos"][: h.size(dim=1) + 1, :]

        a["h"] = np.squeeze(h.detach().numpy())
        return a

    def calculateTuningCurveReliability(self, WAKEactivity, tuning_curves):
        # FAKEactivity = copy.deepcopy(WAKEactivity)
        FAKEactivity = {"state": WAKEactivity["state"]}
        FAKEactivity = self.makeFAKEdata(WAKEactivity, tuning_curves, start_pos=self.start_pos)
        TCreliability = FAKEactivity["TCcorr"]
        return FAKEactivity, TCreliability

    @staticmethod
    def makeFAKEdata(WAKEactivity, tuning_curves, start_pos=1):
        FAKEactivity = {"state": WAKEactivity["state"]}
        position = WAKEactivity["state"]["agent_pos"]
        WAKE_h = WAKEactivity["h"]
        FAKEactivity["h"] = np.zeros_like(WAKEactivity["h"])

        for cell, (k, tuning_curve) in enumerate(tuning_curves.items()):
            if np.isnan(tuning_curve).all():
                continue
            FAKEactivity["h"][:, cell] = tuning_curve[
                position[: WAKE_h.shape[0], 0] - start_pos,
                position[: WAKE_h.shape[0], 1] - start_pos,
            ]

        spaceRemoved = WAKE_h - FAKEactivity["h"]
        EVSpace = 1 - np.var(spaceRemoved, axis=0) / (np.var(WAKE_h, axis=0))
        EVSpace[np.isinf(EVSpace)] = 0
        FAKEactivity["TCcorr"] = EVSpace

        return FAKEactivity

    def cellMapFigure(self):
        plt.figure(figsize=(15, 12))
        plt.subplot(2, 2, 1)
        self.pcaScatterPanel(color=self.groupID, numex=6)

        for i, (metricname, metric) in enumerate(self.metrics.items()):
            plt.subplot(4, 6, i + 13)
            self.pcaScatterPanel(color=metric)
            plt.colorbar(label=metricname, location="bottom")

    def pcaScatterPanel(self, color="black", numex=0, s=25, joint=False):
        if joint:
            data_all = transform_pca(self.joint_metrics, self.joint_pca, self.joint_pca_scaler)
            data_pca = transform_pca(self.metrics, self.joint_pca, self.joint_pca_scaler)
        else:
            data_pca = transform_pca(self.metrics, self.pca, self.pca_scaler)

        if joint:
            plt.scatter(data_all[:, 0], data_all[:, 1], c="gainsboro", s=s)
        plt.scatter(data_pca[:, 0], data_pca[:, 1], c=color, s=s)
        plt.axis("off")

        if numex > 0:
            totalPF = np.array(list(self.tuning_curves.values())).sum(axis=0)
            mask = np.array((totalPF > 0) * 1.0)
            excells = randScatterPoints(data_pca, numex)
            ScatterImages(
                data_pca[excells, 0],
                data_pca[excells, 1],
                [self.tuning_curves[x] for x in excells],
                zoom=1.8,
                mask=mask,
            )

    def tuningCurvepanel(self, excell, inputCell=False, noRecCell=False, title=False):
        place_fields = self.tuning_curves

        totalPF = np.array(list(place_fields.values())).sum(axis=0)
        mask = np.array((totalPF > 0) * 1.0)

        plt.imshow(
            place_fields[excell].transpose(),
            interpolation="nearest",
            alpha=mask.transpose(),
        )
        if title:
            plt.title(f"SI:{self.metrics['SI'][excell]:.2f} EV:{self.metrics['EVs'][excell]:.1f}")
        plt.axis("off")

    def cellGroupExamplesFigure(
        self,
        netname=None,
        savefolder=None,
        groupID=None,
        sortby="SI",
        seed=None,
        dims=(4, 11),
        titles=False,
        fig=None,
    ):
        numex = dims[0] * dims[1]
        if fig is None:
            fig = plt.figure(figsize=(dims[1], dims[0]))

        # GroupID can be an integer or a boolean array
        if isinstance(groupID, int):
            groupcells = np.where(self.groupID == groupID)[0]
        elif groupID is None:
            groupcells = np.where(self.groupID)[0]
        else:
            groupcells = np.where(groupID)[0]

        if seed is not None:
            np.random.seed(seed)
        allexcells = np.random.choice(groupcells, np.min([numex, len(groupcells)]), replace=False)
        if isinstance(sortby, str):
            SIsortinds = self.metrics[sortby][allexcells].argsort()
        else:
            SIsortinds = sortby[allexcells].argsort()
        allexcells = allexcells[SIsortinds]

        for eidx, excell in enumerate(allexcells):
            ax1 = fig.add_subplot(dims[0], dims[1], 1 + convertHorzVertIdx(eidx, dims[0], dims[1]))
            # fg.subplot(4,11,1+convertHorzVertIdx(eidx,4,11))
            self.tuningCurvepanel(excell, title=titles)

        plt.subplots_adjust(wspace=0.2, hspace=0.1)
        if netname is not None:
            saveFig(fig, "cellGroupExamples_" + netname, savefolder, filetype="pdf")
        # plt.show()

    def ClassificationStepFigure(self, ingroup, remaining, xmetric, ymetric, group_name=None):
        fig = plt.figure(figsize=(10, 5))
        subfigs = fig.subfigures(2, 2, wspace=0.07, width_ratios=[1, 3])
        axsLeft = subfigs[0, 0].add_subplot(1, 1, 1)
        axsLeft.plot(
            self.metrics[xmetric][remaining],
            self.metrics[ymetric][remaining],
            ".",
            color="gray",
            markersize=3,
        )
        axsLeft.plot(
            self.metrics[xmetric][ingroup],
            self.metrics[ymetric][ingroup],
            "k.",
            markersize=3,
        )
        # axsLeft.plot(untunedThresh_SI*np.ones(2),[0,untunedThresh_EVs],'r--')
        # axsLeft.plot([0,untunedThresh_SI],untunedThresh_EVs*np.ones(2),'r--')
        axsLeft.set_xlabel(xmetric)
        axsLeft.set_ylabel(ymetric)
        axsLeft.set_title(group_name)

        self.cellGroupExamplesFigure(groupID=ingroup, fig=subfigs[0, 1])
        subfigs[0, 1].suptitle(group_name)
        self.cellGroupExamplesFigure(groupID=remaining, fig=subfigs[1, 1])
        subfigs[1, 1].suptitle("Remaining Cells")

    def cellClassificationFigures(self):
        step1 = self.cellgroups["untuned"] | self.cellgroups["HD_cells"]
        remaining = ~step1
        self.ClassificationStepFigure(step1, remaining, "SI", "EVs", group_name="Untuned/HD Cells")

        step2 = self.cellgroups["border_cells"]
        remaining = ~step2 & ~step1
        self.ClassificationStepFigure(
            step2,
            remaining,
            "fieldasymmetry",
            "border_score",
            group_name="Border Cells",
        )

        step3 = self.cellgroups["single_field"]
        remaining = ~step3 & ~step2 & ~step1
        self.ClassificationStepFigure(
            step3, remaining, "fieldasymmetry", "EVs", group_name="Single Field Cells"
        )

        step4 = self.cellgroups["spatial_HD"]
        remaining = ~step4 & ~step3 & ~step2 & ~step1
        self.ClassificationStepFigure(
            step4, remaining, "SI", "HD_info", group_name="Spatial-HD Cells"
        )

    def cellClassificationFigure(self, netname=None, savefolder=None, withExamples=True):
        numGroups = len(self.cellgroups)
        cmap = plt.get_cmap("viridis", numGroups)

        fig = plt.figure(figsize=(8, 16))
        subfigs = fig.subfigures(numGroups + 1, 1, wspace=0.07)
        topfig = subfigs[0]
        topfig.add_subplot(2, 3, 1)
        cellClassHistogram(self.groupID, groupNames=list(self.cellgroups.keys()), cmap=cmap)

        topfig.add_subplot(1, 3, 2)
        self.pcaScatterPanel(color=self.groupID, s=3)

        topfig.add_subplot(1, 3, 3)
        plt.scatter(self.metrics["SI"], self.metrics["EVs"], c=self.groupID, s=3)
        plt.xlabel("SI")
        plt.ylabel("EVs")

        if withExamples:
            for gidx, groupname in enumerate(self.cellgroups.keys()):
                self.cellGroupExamplesFigure(groupID=gidx, fig=subfigs[gidx + 1])
                subfigs[gidx + 1].suptitle(groupname)

        plt.tight_layout()

        if netname is not None:
            saveFig(fig, "CellTypes_" + netname, savefolder, filetype="pdf")

        plt.show()


def FitJointPCA(TCAlist):
    # Concatenate all metrics
    # joint_metrics
    joint_metrics = {}
    for key, value in TCAlist[0].metrics.items():
        joint_metrics[key] = np.concatenate([TCA.metrics[key] for TCA in TCAlist], axis=0)

    for TCA in TCAlist:
        TCA.joint_metrics = joint_metrics
        TCA.joint_pca, TCA.joint_pca_scaler = fit_pca(TCA.joint_metrics)
    return joint_metrics


def LRoomWallGroups(env):
    allwalls = find_walls(env.env.grid)
    near_walls, not_near_walls = get_near_walls(allwalls)

    topwall = find_walls(env.env.grid, rowmax=1)
    near_topwall, _ = get_near_walls(topwall, exclude=allwalls)

    leftwall = find_walls(env.env.grid, colmax=1)
    near_leftwall, _ = get_near_walls(leftwall, exclude=allwalls)

    rightwall = find_walls(env.env.grid, colmin=env.env.grid.height - 1)
    near_rightwall, _ = get_near_walls(rightwall, exclude=allwalls)

    bottomwall = find_walls(env.env.grid, rowmin=env.env.grid.width - 1)
    near_bottomwall, _ = get_near_walls(bottomwall, exclude=allwalls)

    innerwalls = find_walls(
        env.env.grid,
        rowmin=1,
        rowmax=env.env.grid.width - 2,
        colmin=1,
        colmax=env.env.grid.height - 2,
    )
    near_innerwall, _ = get_near_walls(innerwalls, exclude=allwalls)

    wallgroups = [
        near_topwall,
        near_leftwall,
        near_rightwall,
        near_bottomwall,
        near_innerwall,
    ]
    return wallgroups, not_near_walls


def find_walls(grid, rowmin=None, rowmax=None, colmin=None, colmax=None):
    if rowmin is None:
        rowmin = 0
    if rowmax is None:
        rowmax = grid.width
    if colmin is None:
        colmin = 0
    if colmax is None:
        colmax = grid.height

    walls = np.zeros((grid.width, grid.height))
    for i in range(rowmin, rowmax):
        for j in range(colmin, colmax):
            grid_at_loc = grid.get(i, j)
            if grid_at_loc is not None and grid_at_loc.type == "wall":
                walls[i, j] = 1
    return walls


def get_near_walls(walls, distance_threshold=2, exclude=None):
    distances = distance_transform_edt(~(walls == 1))
    distances = distances[1:-1, 1:-1]

    near_walls = (distances < distance_threshold) & (distances > 0)
    not_near_walls = distances > distance_threshold

    if exclude is not None:
        near_walls = near_walls & ~(exclude[1:-1, 1:-1] == 1)
        not_near_walls = not_near_walls & ~(exclude[1:-1, 1:-1] == 1)

    return near_walls, not_near_walls


def calculateOneBorderScore(tuning_curve, near_walls, not_near_walls):
    mean_border_rate = np.mean(tuning_curve * near_walls)
    mean_nonborder_rate = np.mean(tuning_curve * not_near_walls)
    border_score = (mean_border_rate - mean_nonborder_rate) / (
        mean_border_rate + mean_nonborder_rate
    )
    return border_score


def calculateBorderScore(tuning_curve, near_walls, not_near_walls):
    border_scores = []
    for i in near_walls:
        border_scores.append(calculateOneBorderScore(tuning_curve, i, not_near_walls))
    return max(border_scores)


def pf_autocorr(d, meanNorm=False, peakNorm=False):
    """
    Compute the 2D autocorrelation of each matrix in a dictionary and return a new dictionary with the autocorrelated matrices.
    """
    d_with_autocorr = {}
    for key in d:
        matrix = d[key]
        if meanNorm:
            matrix = matrix / (np.mean(matrix, axis=None))
        autocorr = correlate2d(matrix, matrix, mode="full")
        if peakNorm:
            autocorr = autocorr / np.max(autocorr, axis=None)
        d_with_autocorr[key] = autocorr
    return d_with_autocorr


def pfdict_to_np(pfdict):
    """
    Convert the items of a dictionary to a NumPy tensor.
    Each item is assumed to be a matrix, which will be flattened and become a row of the tensor.
    """
    rows = []
    for key in pfdict:
        matrix = pfdict[key]
        rows.append(matrix.flatten())
    tensor = np.vstack(rows)
    return tensor


def count_autocorr_peaks(autocorr, size=3, threshold=0.15):
    autocorr_norm = autocorr
    local_max = (maximum_filter(autocorr_norm, size=size) == autocorr_norm) & (
        autocorr_norm > threshold
    )
    labeled, num_features = label(local_max)
    return num_features


def calculate_field_size(tc_autocorr, threshold=0.5):
    field = tc_autocorr > threshold
    labeled, num_features = label(field)
    if num_features == 0:
        print("dead cell")
        return np.nan
    whicharea = labeled[tc_autocorr == np.max(tc_autocorr, axis=None)]
    centerlabeled = labeled == whicharea
    area = np.sqrt(np.sum(centerlabeled))
    return area


def calculate_field_asymmetry(tc_autocorr, threshold=0.5):
    field = tc_autocorr > threshold
    labeled, num_features = label(field)
    if num_features == 0:
        print("dead cell")
        return np.nan
    whicharea = labeled[tc_autocorr == np.max(tc_autocorr, axis=None)]
    centerlabeled = labeled == whicharea

    coords = np.column_stack(np.nonzero(centerlabeled))
    yc, xc = field.shape[0] // 2, field.shape[1] // 2
    coords = coords - [yc, xc]
    # centerlabeled.shape

    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

    # Draw the ellipse
    major_axis, minor_axis = 2 * np.sqrt(eigvals)
    minor_axis = np.max([minor_axis, 1])

    return major_axis / minor_axis


def fit_pca(metrics):
    data = np.vstack(list(metrics.values())).T

    data_scaler = StandardScaler()
    data_scaler.fit(data)
    data_standardized = data_scaler.transform(data)
    data_standardized[np.isnan(data_standardized)] = 0

    # Fit the PCA model
    pca = PCA(n_components=2)  # Number of principal components to keep
    pca.fit(data_standardized)

    return pca, data_scaler


def transform_pca(metrics, pca, data_scaler):
    data = np.vstack(list(metrics.values())).T
    data_standardized = data_scaler.transform(data)
    data_standardized[np.isnan(data_standardized)] = 0
    return pca.transform(data_standardized)


def ScatterImages(x, y, img, zoom=1.5, mask=1):
    ax = plt.gca()
    for i, xval in enumerate(x):
        ab = AnnotationBbox(getImage(img[i], zoom=zoom, mask=mask), (x[i], y[i]), frameon=False)
        ax.add_artist(ab)


def getImage(img, zoom=1.5, mask=1):
    return OffsetImage(img, zoom=zoom, alpha=0.8 * mask)


def randScatterPoints(xy, n):
    # Calculate the range of the x and y values
    x = xy[:, 0]
    y = xy[:, 1]

    # Divide the range of the x and y values into n equal intervals
    x_intervals = np.linspace(np.min(x), np.max(x), n + 1)
    y_intervals = np.linspace(np.min(y), np.max(y), n + 1)

    # Randomly select one point from each interval
    x_samples = []
    y_samples = []
    for i in range(n):
        for j in range(n):
            x_mask = np.logical_and(x >= x_intervals[i], x <= x_intervals[i + 1])
            y_mask = np.logical_and(y >= y_intervals[j], y <= y_intervals[j + 1])
            mask = np.logical_and(x_mask, y_mask)
            indices = np.where(mask)[0]
            if len(indices) > 0:
                idx = np.random.choice(indices)
                x_samples.append(x[idx])
                y_samples.append(y[idx])

    # Reshape the randomly selected points into a 2D array
    samples = np.vstack((x_samples, y_samples)).T

    # Find the indices of the selected points in the original data
    indices = []
    for i, du in enumerate(x_samples):
        idx = np.where((x == x_samples[i]) & (y == y_samples[i]))[0][0]
        indices.append(idx)

    return np.array(indices)


def convertHorzVertIdx(i, h, w):
    row = np.mod(i, h)
    col = int(np.floor(i / h))
    j = row * w + col
    return j


def cellClassHistogram(groupID, groupNames=None, colors=None, cmap=None):
    # Create bins for ALL possible groups (not just present ones)
    max_group = max(groupID) if groupNames is None else len(groupNames) - 1
    counts, bins, patches = plt.hist(groupID, bins=np.arange(-0.5, max_group + 1.5), density=True)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    if groupNames is not None:
        plt.xticks(bin_centers, groupNames, rotation=90)

    if cmap is not None:
        # Color ALL patches, including zero-count ones
        for gidx, patch in enumerate(patches):
            color = cmap(gidx)
            patch.set_facecolor(color)
