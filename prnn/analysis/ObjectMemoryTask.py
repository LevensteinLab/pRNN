from prnn.utils.env import make_env
import copy
from prnn.utils.agent import RandomActionAgent
import numpy as np
import matplotlib.pyplot as plt
from prnn.utils.general import saveFig





class ObjectMemoryTask:
    def __init__(self, predictiveNet,  decoder='train',
                num_trials=100, trial_duration=1000, lr_trials=2, lr_groups = [0,1,2]):
        
        self.pN = predictiveNet
        
        self.env_object = self.makeObjectEnvironment()
        self.env = self.pN.EnvLibrary[0]
        self.goal_loc = self.env_object.env.env.env.goal_pos
        
        #Train the decoder 
        #if decoder is 'train':
        #    self.decoder = self.trainDecoder()
        #else:
        #    self.decoder = decoder
        
        #Train the network in the object room #Note, copy ExperienceReplayAnalysis 
        #approach for multiple learning rates (consider rec and FF learning rates)
        self.pN_post = self.trainNovelObject(self.pN, self.env_object, 
                                            num_trials=num_trials, 
                                            sequence_duration = trial_duration, 
                                            lr_trials=lr_trials, lrgroups=lr_groups)
        
        self.testTrial = self.getTestTrial(self.pN_post,self.pN,self.env)
        self.objectLearning = self.quantifyObjectLearning(self.testTrial)
        
        
        

    #def lr_panel
    
    
    def makeObjectEnvironment(self):
        env_object_name = 'MiniGrid-LRoom_Goal-18x18-v0'
        env_object = make_env(env_object_name)
        return env_object
    
    
    def trainDecoder(self):
        env = self.pN.EnvLibrary[0]
        action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        agent = RandomActionAgent(env.action_space,action_probability)
        _, _, decoder = self.pN.calculateSpatialRepresentation(env,
                                                               agent,
                                                               numBatches=10000,
                                                               trainDecoder=True)
        return decoder
    
    
    def trainNovelObject(self, pN, env_object, num_trials=100,
                         sequence_duration = 1000, 
                         lr_trials=2, lrgroups=[0,1,2],
                         resetOptimizer=False, continueTraining=False):
        if continueTraining:
            print('continuing training')
            pN_post = pN
        else:
            pN_post = copy.deepcopy(pN)
            
        if resetOptimizer:
            print('resetting optimizer')
            pN_post.resetOptimizer(pN_post.trainArgs.lr,
                                   pN_post.trainArgs.weight_decay)

        action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        agent = RandomActionAgent(env_object.action_space,action_probability)
        
        # print(lr_trials)
        #Update the learning rate
        oldlr = [0. for i in lrgroups]
        if isinstance(lr_trials, int):
            lr_trials = [lr_trials for i in lrgroups]
        for lidx,lgroup in enumerate(lrgroups):
            oldlr[lidx] = pN_post.optimizer.param_groups[lgroup]['lr']
            pN_post.optimizer.param_groups[lgroup]['lr'] = oldlr[lidx] * lr_trials[lidx]

        pN_post.trainingEpoch(env_object, agent, num_trials=num_trials,
                              sequence_duration = sequence_duration,
                              learningRate = None)

        #Return the learning rate
        for lidx, lgroup in enumerate(lrgroups):
            pN_post.optimizer.param_groups[lgroup]['lr'] = oldlr[lidx]

        return pN_post
    
    
    def getTestTrial(self, pN_post, pN_control, env, timesteps=2500,
                        ):
        # env = self.pN.EnvLibrary[0]
        action_probability = np.array([0.15,0.15,0.6,0.1,0,0,0])
        agent = RandomActionAgent(env.action_space,action_probability)
        
        #Collect observation sequence in the environment    
        obs,act,state,render = pN_post.collectObservationSequence(env,agent,
                                                                  timesteps, 
                                                                  includeRender=True)
        
        #Predict observations with the trained and untrained nets
        obs_pred, obs_next, _ = pN_post.predict(obs,act)
        obs_pred_notrain, _, _ = pN_control.predict(obs,act)
        
        objectTest = {
            'obs' : obs,
            'obs_pred' : obs_pred,
            'obs_pred_control' : obs_pred_notrain,
            'state' : state,
            'render' : render,
        }
        return objectTest
        
        
    def quantifyObjectLearning(self, objectTest, control_location=[2,7], whichPhase=0):
        pos = objectTest['state']['agent_pos'][whichPhase:,:]
        HD = objectTest['state']['agent_dir'][whichPhase:]
        obs_pred =  objectTest['obs_pred']
        obs_pred_notrain =  objectTest['obs_pred_control']
        goal_loc = self.goal_loc
        
        #Get the predicted pixel values at the object/control location
        obs_np = self.pN_post.env_shell.pred2np(obs_pred,whichPhase=whichPhase)
        locobs, inviewtimes, viewcoords = get_obs_at_loc(obs_np,goal_loc,pos,HD)
        conobs, _, _ = get_obs_at_loc(obs_np,control_location,pos,HD)
        
        #Get the predicted pixel values in the control networks
        obs_np = self.pN_post.env_shell.pred2np(obs_pred_notrain,whichPhase=whichPhase)
        locobs_notrain, inviewtimes, viewcoords = get_obs_at_loc(obs_np,goal_loc,pos,HD)
        conobs_notrain, _, _ = get_obs_at_loc(obs_np,control_location,pos,HD)
        
        objectloc_deltaobs = locobs-locobs_notrain
        controlloc_deltaobs = conobs-conobs_notrain
        
        goalmodulation = np.mean(objectloc_deltaobs[:,1])
        ctlmodulation_diffcolor = np.mean(np.concatenate((objectloc_deltaobs[:,0],objectloc_deltaobs[:,2])))
        ctlmodulation_diffloc = np.mean(controlloc_deltaobs[:,1])
        
        
        objectLearning = {
            'inviewtimes' : inviewtimes,
            'viewcoords' : viewcoords,
            'objectloc_obs' : locobs,
            'controlloc_obs' : conobs,
            'objectloc_obs_controlNet' : locobs_notrain,
            'controlloc_obs_controlNet' : conobs_notrain,
            'objectloc_deltaobs' : objectloc_deltaobs,
            'controlloc_deltaobs' : controlloc_deltaobs,
            'goalmodulation' : goalmodulation,
            'ctlmodulation_diffcolor' : ctlmodulation_diffcolor,
            'ctlmodulation_diffloc' : ctlmodulation_diffloc,
        }
        return objectLearning
        
        
    def ObjectLearningFigure(self, netname=None, savefolder=None,
                            whichview=1):
        
        
        plt.subplot(4,3,1)
        self.objPixelChangePanel(self.objectLearning)
        
        self.exampleObsSequencePanel(self.testTrial,self.objectLearning,
                                    whichview=whichview)
        
        plt.tight_layout()
        if netname is not None:
            saveFig(plt.gcf(),'ObjectLearning_'+netname,savefolder,
                    filetype='pdf')
        plt.show()
        
        
    def objPixelChangePanel(self, objectLearning):
        deltaobs_goal = objectLearning['objectloc_deltaobs']
        deltaobs_ctl = objectLearning['controlloc_deltaobs']
        
        plt.boxplot(deltaobs_goal, showfliers=False, positions = [1.8,1,2.2], 
                    labels = ['R','G','B'])
        plt.boxplot(deltaobs_ctl, showfliers=False, positions = [4.3,3.5,4.7], 
                    labels = ['R','G','B'])
        plt.plot(plt.xlim(),[0,0],'k--')
        plt.plot([3,3],plt.ylim(),'k:')
        plt.ylabel('Change in Predicted Observation')
        
    
    def exampleObsSequencePanel(self,objectTest, objectLearning, firstrow=3, whichview=1):
        render = objectTest['render']
        obs = objectTest['obs']
        obs_pred = objectTest['obs_pred']
        obs_pred_notrain = objectTest['obs_pred_control']
        pN = self.pN
        
        inviewtimes = self.objectLearning['inviewtimes']
        extimes = range(inviewtimes[whichview]-2,inviewtimes[whichview]+5)
        
        pN.plotSequence(render,
                          extimes,firstrow,label='State')

        pN.plotSequence(pN.env_shell.pred2np(obs),
                          extimes,firstrow+1,label='Obs')

        pN.plotSequence(pN.env_shell.pred2np(obs_pred_notrain),
                          extimes,firstrow+2,label='Pred_CTL')
        
        pN.plotSequence(pN.env_shell.pred2np(obs_pred),
                          extimes,firstrow+3,label='Pred')


        
        

        
        
        
# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]

def get_view_coords(i, j, pos, HD, agent_view_size=7):
    ax, ay = pos
    dx, dy = DIR_TO_VEC[HD]
    rx, ry = -dy, dx

    # Compute the absolute coordinates of the top-left view corner
    sz = agent_view_size
    hs = agent_view_size // 2
    tx = ax + (dx * (sz-1)) - (rx * hs)
    ty = ay + (dy * (sz-1)) - (ry * hs)

    lx = i - tx
    ly = j - ty

    # Project the coordinates of the object relative to the top-left
    # corner onto the agent's own coordinate system
    vx = (rx*lx + ry*ly)
    vy = -(dx*lx + dy*ly)

    return vx, vy

def get_obs_at_loc(obs, goal_loc, pos, HD):
    i ,j = goal_loc
    
    locobs=[]
    viewtimes = []
    viewcoords = []
    for tt in range(obs.shape[0]):
        vx, vy = get_view_coords(i,j, pos[tt,:], HD[tt])
        if (vx >= 0) & (vx < 7) & (vy >= 0) & (vy < 7):
            locobs.append(obs[tt,vy,vx,:])
            viewtimes.append(tt)
            viewcoords.append([vx, vy])
    
    locobs = np.stack(locobs,axis=0)
    viewtimes = np.stack(viewtimes,axis=0)
    viewcoords = np.stack(viewcoords,axis=0)
    return locobs, viewtimes, viewcoords