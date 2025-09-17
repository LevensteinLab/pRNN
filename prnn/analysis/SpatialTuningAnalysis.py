from prnn.utils.agent import RandomActionAgent
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from prnn.utils.general import saveFig, savePkl, loadPkl


class SpatialTuningAnalysis:
    def __init__(
        self,
        predictiveNet,
        timesteps_wake=5000,
        inputControl=False,
        untrainedControl=False,
        reliabilityMetric="EVspace",
        compareNoRec=False,
        ratenorm=True,
        activeTimeThreshold=250,
        agent=False,
        theta="expand",
        fig_type="png",
    ):
        self.pN = predictiveNet
        self.inputControl = inputControl
        self.untrainedControl = untrainedControl
        self.noRec = compareNoRec
        self.reliabilityMetric = reliabilityMetric
        self.fig_type = fig_type

        self.env = predictiveNet.EnvLibrary[0]
        self.start_pos = (
            self.env.start_pos
        )  # the numbering of occupiable locations starts from this

        # this is for backward compatibility, better provide the agent as an arg
        if not agent:
            action_probability = np.array([0.15, 0.15, 0.6, 0.1, 0, 0, 0])
            agent = RandomActionAgent(self.env.action_space, action_probability)

        if self.inputControl:
            print("Getting Tuning Curves for Input Units")
            place_fields, SI, _ = self.pN.calculateSpatialRepresentation(
                self.env,
                agent,
                timesteps=max(15000, timesteps_wake),
                inputControl=True,
                bitsec=not (ratenorm),
                activeTimeThreshold=activeTimeThreshold,
            )
            self.SI = SI["SI"].values
            self.inputFields = SI["inputFields"]
            self.inputSI = SI["inputCtrl"][~np.isnan(SI["inputCtrl"])]
            self.tuning_curves = place_fields
        else:
            try:  # Bug with nets that don't save whole training data
                self.tuning_curves = self.pN.TrainingSaver["place_fields"].values[-1]
                self.SI = np.squeeze(self.pN.TrainingSaver["SI"].values[-1])
            except:
                self.tuning_curves = self.pN.TrainingSaver["place_fields"]
                self.SI = self.pN.TrainingSaver["SI"]

        if self.untrainedControl:
            print("Running Untrained Control")
            self.pNControl = makeUntrainedNet(
                self.pN, self.env, agent, ratenorm=ratenorm
            )
            self.untrainedFields = self.pNControl.TrainingSaver["place_fields"].values[
                -1
            ]
            self.untrainedSI = self.pNControl.TrainingSaver["SI"].values[-1]
            WAKEactivity = self.runWAKE(
                self.pNControl, self.env, agent, timesteps_wake, theta=theta
            )
            FAKEuntraineddata = self.makeFAKEdata(
                WAKEactivity,
                self.untrainedFields,
                start_pos=self.start_pos,
                n_obs=self.b_obs,
            )
            self.untrainedReliability = FAKEuntraineddata["TCcorr"]

        # Calculate TC reliability
        # Run WAKE
        self.WAKEactivity = self.runWAKE(
            self.pN, self.env, agent, timesteps_wake, theta=theta
        )
        print("Calculating EV_s")
        self.FAKEactivity, self.TCreliability = self.calculateTuningCurveReliability(
            self.WAKEactivity, self.tuning_curves
        )

        if inputControl:
            print("Calculating EV_s for input control")
            FAKEinputdata = self.makeFAKEdata(
                self.WAKEactivity,
                self.inputFields,
                inputCells=True,
                start_pos=self.start_pos,
                n_obs=self.b_obs,
            )
            self.inputReliability = FAKEinputdata["TCcorr"]

        # Compare to a Recurrence-ablated control
        if self.noRec:
            pN_noRec = self.makeNoRecNet(self.pN)
            self.noRec_fields, _, _ = pN_noRec.calculateSpatialRepresentation(
                self.env, agent, bitsec=not (ratenorm)
            )
            self.reccorr = self.noRecComparison(self.tuning_curves, self.noRec_fields)

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
        FAKEactivity = self.makeFAKEdata(
            WAKEactivity, tuning_curves, start_pos=self.start_pos, n_obs=self.b_obs
        )
        TCreliability = FAKEactivity["TCcorr"]
        return FAKEactivity, TCreliability

    def makeNoRecNet(self, pN):
        from torch import no_grad

        pN_noRec = pN.copy()
        # pN_noRec.pRNN.W.requres_grad = False
        with no_grad():
            pN_noRec.pRNN.W.subtract_(pN_noRec.pRNN.W)
            pN_noRec.pRNN.W_in[:, -pN_noRec.act_size :].subtract_(
                pN_noRec.pRNN.W_in[:, -pN_noRec.act_size :]
            )
            # pN_noRec.pRNN.W.add_(torch.eye(pN_noRec.hidden_size).mul_(1-1/pN_noRec.pRNN.neuralTimescale))

        return pN_noRec

    def noRecComparison(self, place_fields, noRec_fields):
        reccorr = np.ones(len(place_fields.keys()))
        PFmask = np.array(list(place_fields.values())).sum(axis=0)
        PFmask_idx = np.nonzero(PFmask > 0)
        for tcidx, (tc_key, tc) in enumerate(place_fields.items()):
            reccorr[tcidx], _ = spearmanr(
                tc[PFmask_idx], noRec_fields[tc_key][PFmask_idx]
            )

        return reccorr

    @staticmethod
    def makeFAKEdata(
        WAKEactivity,
        tuning_curves,
        useMstats=False,
        metric="EVspace",
        inputCells=False,
        start_pos=1,
        n_obs=1,
    ):
        FAKEactivity = {"state": WAKEactivity["state"]}
        position = WAKEactivity["state"]["agent_pos"]
        if inputCells:
            if n_obs == 1:
                WAKE_h = WAKEactivity["obs"].squeeze().detach().numpy()[:-1, :]
            else:
                WAKE_h = np.concatenate(
                    [o.squeeze().detach().numpy()[:-1, :] for o in WAKEactivity["obs"]],
                    axis=-1,
                )
            FAKEactivity["h"] = np.zeros_like(WAKE_h)
        else:
            WAKE_h = WAKEactivity["h"]
            FAKEactivity["h"] = np.zeros_like(WAKEactivity["h"])

        for cell, (k, tuning_curve) in enumerate(tuning_curves.items()):
            if np.isnan(tuning_curve).all():
                continue
            FAKEactivity["h"][:, cell] = tuning_curve[
                position[: WAKE_h.shape[0], 0] - start_pos,
                position[: WAKE_h.shape[0], 1] - start_pos,
            ]

        if metric == "EVspace":
            spaceRemoved = WAKE_h - FAKEactivity["h"]
            EVSpace = 1 - np.var(spaceRemoved, axis=0) / (np.var(WAKE_h, axis=0))
            EVSpace[np.isinf(EVSpace)] = 0
            FAKEactivity["TCcorr"] = EVSpace

        elif metric == "TCcorr":
            # Calculate the tuning-curve reliability
            # add small random amount
            adjusted_wake = WAKE_h
            adjusted_fake = FAKEactivity["h"]
            FAKEcorr = spearmanr(adjusted_wake, adjusted_fake, axis=0)
            if (
                useMstats
                and FAKEcorr.correlation.size == 1
                and np.isnan(FAKEcorr.correlation).all()
            ):
                print(FAKEcorr.correlation)
                print("correlation nan, using mstats (slower)")
                FAKEcorr = spearmanr(WAKE_h, FAKEactivity["h"], axis=0)
                print(FAKEcorr.correlation)
            Nneurons = np.size(WAKE_h, 1)
            FAKEcorr = np.diagonal(FAKEcorr.correlation, offset=Nneurons)
            FAKEactivity["TCcorr"] = FAKEcorr

        return FAKEactivity

    def TCSIpanel(self, excells=None, incInput=True):
        SI = self.SI
        TCreliability = self.FAKEactivity["TCcorr"]

        if hasattr(self, "untrainedReliability"):
            plt.plot(
                self.untrainedSI,
                self.untrainedReliability,
                ".",
                markersize=2,
                label="untrained",
                color="grey",
            )
        if hasattr(self, "inputReliability") and incInput:
            plt.plot(
                self.inputSI,
                self.inputReliability,
                ".",
                markersize=2,
                label="input",
                color="red",
            )
            plt.legend()
        plt.plot(SI, TCreliability, "k.", markersize=2, label="pRNN units")
        for excell in excells:
            plt.plot(
                SI[excell], TCreliability[excell], "o", color="grey", fillstyle="none"
            )

        plt.xlabel("SI")
        plt.ylabel("EV Space")

        plt.ylabel(self.reliabilityMetric)
        plt.ylim([-0.1, 1.1])

    def TCreliabilitypanel(self, excell):
        plt.plot(
            self.WAKEactivity["h"][:, excell],
            self.FAKEactivity["h"][:, excell],
            "k.",
            markersize=1,
        )
        bound = np.min([plt.xlim()[1], plt.ylim()[1]])

        plt.plot([0, bound], [0, bound], "k--")
        plt.xlabel("real rate")
        plt.ylabel("Fake Rate")

    def tuningCurvepanel(self, excell, inputCell=False, noRecCell=False, title=True):
        place_fields = self.tuning_curves
        SI = self.SI
        EV = self.TCreliability
        totalPF = np.array(list(place_fields.values())).sum(axis=0)
        mask = np.array((totalPF > 0) * 1.0)
        if inputCell:
            place_fields = self.inputFields
            SI = self.inputSI
            EV = self.inputReliability
        if noRecCell:
            place_fields = self.noRec_fields

        plt.imshow(
            place_fields[excell].transpose(),
            interpolation="nearest",
            alpha=mask.transpose(),
        )
        if title:
            plt.title(f"SI:{SI[excell]:.2f} EV:{EV[excell]:.1f}")
        plt.axis("off")

    def TCReliabilityFigure(
        self,
        netname=None,
        savefolder=None,
        excells=None,
        threshold=0.6,
        numex=18,
        seed=None,
    ):
        # Calculate RGA.SIdep = RGA.calculateSIdependence() first... This is sloppy

        fg = plt.figure(figsize=(18, 12))

        # Panel of High Reliability and low reliability cells (highest SI)
        # Each panel in a subfigure...
        reliablecells = np.nonzero(self.TCreliability > threshold)[0]
        unreliablecells = np.nonzero(self.TCreliability < threshold - 0.2)[0]

        if seed is not None:
            np.random.seed(seed)
        allexcells = np.random.choice(reliablecells, numex, replace=False)
        SIsortinds = self.SI[allexcells].argsort()
        allexcells = allexcells[SIsortinds]

        if excells is None:
            # excells = [reliablecells[1],unreliablecells[1]]
            excells = [unreliablecells[0], reliablecells[0], reliablecells[2]]

        plt.subplot(3, 3, 1)
        self.TCSIpanel(excells, incInput=False)

        for eidx, excell in enumerate(excells):
            plt.subplot(6, 6, 4 + eidx)
            self.tuningCurvepanel(excell)

            plt.subplot(6, 6, 10 + eidx)
            self.TCreliabilitypanel(excell)
            if eidx > numex:
                break

        plt.ylabel(None)

        for eidx, excell in enumerate(allexcells):
            plt.subplot(6, 6, 13 + convertHorzVertIdx(eidx, 4, 6))
            self.tuningCurvepanel(excell)
            if eidx > numex:
                break

        # for idx,whichcells in enumerate([reliablecells[0:40],unreliablecells]):
        #    subfig = plt.subplot(2,2,2+idx*2)
        #    #plt.suptitle('Example reliable cells')

        # subfg = fg.add_subfigure(subfig.get_subplotspec())
        # self.pN.plotTuningCurvePanel(SI=True,whichcells=whichcells,
        #                              fig = subfg,gridsize=4)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        if netname is not None:
            saveFig(fg, "TCReliability_" + netname, savefolder, filetype=self.fig_type)
        # plt.show()

    def TCExamplesFigure(
        self,
        netname=None,
        savefolder=None,
        numex=32,
        seed=None,
        EVthresh=0.5,
        sortby="SI",
        excells=None,
    ):
        # Calculate RGA.SIdep = RGA.calculateSIdependence() first... This is sloppy

        fg = plt.figure(figsize=(18, 7))

        if excells is None:
            reliablecells = np.nonzero(self.TCreliability > EVthresh)[0]
        else:
            reliablecells = excells

        if seed is not None:
            np.random.seed(seed)
        allexcells = np.random.choice(
            reliablecells, np.min([numex, len(reliablecells)]), replace=False
        )
        if sortby == "SI":
            SIsortinds = self.SI[allexcells].argsort()
        else:
            SIsortinds = sortby[allexcells].argsort()
        allexcells = allexcells[SIsortinds]

        plt.subplot(2, 4, 8)
        self.TCSIpanel(allexcells, incInput=False)
        plt.plot(plt.xlim(), EVthresh * np.ones(2), "k--")

        for eidx, excell in enumerate(allexcells):
            plt.subplot(4, 11, 1 + convertHorzVertIdx(eidx, 4, 11))
            self.tuningCurvepanel(excell)

        plt.subplots_adjust(wspace=0.2, hspace=0.1)
        if netname is not None:
            saveFig(fg, "TCExamples_" + netname, savefolder, filetype=self.fig_type)
        # plt.show()

    def SpatialTuningFigure(self, netname=None, savefolder=None, exgrid=4, seed=None):
        fg = plt.figure(figsize=(12, 8))

        if seed is not None:
            np.random.seed(seed)

        excells = randScatterPoints(np.array([self.SI, self.TCreliability]).T, exgrid)
        exinputs = [0, 10, 20]

        plt.subplot(3, 3, 7)
        self.TCSIpanel(excells)

        for eidx, excell in enumerate(excells):
            plt.subplot(6, 7, 1 + eidx)
            self.tuningCurvepanel(excell)

            # plt.subplot(5,5,11+eidx)
            # self.TCreliabilitypanel(excell)
        if self.inputControl:
            for eidx, excell in enumerate(exinputs):
                plt.subplot(5, 5, 23 + eidx)
                self.tuningCurvepanel(excell, inputCell=True)

            # plt.subplot(5,5,11+eidx)
            # self.TCreliabilitypanel(excell)

        plt.ylabel(None)

        plt.subplots_adjust(wspace=0.2, hspace=0.3)
        if netname is not None:
            saveFig(fg, "SpatialTuning_" + netname, savefolder, filetype=self.fig_type)
        # plt.show()

    def RecAblationFigure(self, netname=None, savefolder=None):
        placecells = np.where(self.TCreliability > 0.5)[0]
        exgrid = 4

        excells = randScatterPoints(
            np.array([self.SI[placecells], self.reccorr[placecells]]).T, exgrid
        )
        excells = placecells[excells]

        plt.figure()
        plt.subplot(3, 3, 1)
        plt.plot(self.SI[placecells], self.reccorr[placecells], ".")

        plt.subplot(3, 3, 2)
        plt.plot(self.TCreliability, self.reccorr, ".")

        counter = 10
        for eidx, excell in enumerate(excells):
            counter += 1
            plt.subplot(6, 6, counter)
            self.tuningCurvepanel(excell, title=False)
            counter += 1
            plt.subplot(6, 6, counter)
            self.tuningCurvepanel(excell, noRecCell=True, title=False)
            # counter+=1
        # plt.show()

    def saveAnalysis(self, savename, savefolder=None):
        # TODO: default to an analysis folder in pN.savefolder
        savename = savename + "_SpatialTuningAnalysis"
        savePkl(self, savename, savefolder)
        print("Analysis Saved to pathname")

    def loadAnalysis(savename, savefolder=None, suppressText=False):
        # TODO: conver to general.loadPkl
        savename = savename + "_SpatialTuningAnalysis"
        analysis = loadPkl(savename, savefolder)
        if not suppressText:
            print("Analysis Loaded from pathname")
        return analysis


def makeUntrainedNet(
    pN, env, agent, ratenorm=True, decodeError=False, calculatesRSA=False
):
    pNControl = pN.copy()
    pNControl.pRNN.__init__(
        pNControl.obs_size,
        pNControl.act_size,
        pNControl.hidden_size,
        neuralTimescale=pNControl.trainArgs.ntimescale,
        dropp=pNControl.trainArgs.dropout,
        f=pNControl.trainArgs.sparsity,
    )
    _, _, decoder = pNControl.calculateSpatialRepresentation(
        env,
        agent,
        saveTrainingData=True,
        trainDecoder=decodeError,
        calculatesRSA=calculatesRSA,
        bitsec=not (ratenorm),
    )
    if decodeError:
        pNControl.calculateDecodingPerformance(
            env, agent, decoder, saveTrainingData=True, showFig=False
        )
    return pNControl


def randScatterPoints(xy, n):
    # Calculate the range of the x and y values
    x = xy[:, 0]
    y = xy[:, 1]

    # Divide the range of the x and y values into n equal intervals
    x_intervals = np.linspace(np.nanmin(x), np.nanmax(x), n + 1)
    y_intervals = np.linspace(np.nanmin(y), np.nanmax(y), n + 1)

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

    # Sort by the x values
    x_samples, y_samples = zip(*sorted(zip(x_samples, y_samples)))
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
