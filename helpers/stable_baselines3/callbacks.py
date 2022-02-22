class HistoryCallback(BaseCallback):
    """ Save history from MOnitor-s into the model history attribute
    """

    def __init__(self,update_interval=1000):
        super().__init__()
        self.update_interval=update_interval
    
    def _on_training_start(self):
        if not hasattr(self.model,'history'):
            self.model.history={}
        self.train_env=self.model.env
        self.eval_env=None
        callback=self.locals['callback']
        for cb in callback.callbacks if isinstance(callback,CallbackList) or isinstance(callback,list) else [callback]:
            if isinstance(cb,EvalCallback):
                self.eval_env=cb.eval_env
                self.n_eval_episodes=cb.n_eval_episodes
                self.eval_freq=cb.eval_freq
                break
    
    def _on_step(self):    
        if self.num_timesteps % self.update_interval ==0:
            self.update_history_from_monitor()
        return True

    def _on_training_end(self):
        self.update_history_from_monitor()

    def update_history_from_monitor(self):
        if self.model.env is not None:
            lengths=self.model.env.env_method('get_episode_lengths')
            rewards=self.model.env.env_method('get_episode_rewards')
            times=self.model.env.env_method('get_episode_times')
            n=len(lengths)
            l,r,t,s=[],[],[],[]
            for i in range(n):
                l.extend(lengths[i])
                t.extend(times[i])
                r.extend(rewards[i])                
                s.extend((np.array(lengths[i]).cumsum()*n).tolist())            
            if 'train' not in self.model.history or len(l)>0:
                self.model.history['train']=pd.DataFrame({'Lengths':l,'Times':t,'Rewards':r,'CumSteps':s})
            if self.eval_env is not None:
                l,r,t,s=[],[],[],[]
                r=np.array(self.eval_env.env_method('get_episode_rewards')).flatten()
                l=np.array(self.eval_env.env_method('get_episode_lengths')).flatten()
                t=np.array(self.eval_env.env_method('get_episode_times')).flatten()
                s=(np.arange(r.shape[0])//self.n_eval_episodes+1)*self.eval_freq*n-1
                if 'eval' not in self.model.history or len(l)>0:
                    self.model.history['eval']=pd.DataFrame({'Lengths':l,'Times':t,'Rewards':r,'CumSteps':s})

class ProgressCallback(BaseCallback):
    """ Progressbar callback for stable baselines 3 RL algorithms
        There are two types of progress bar:
            tqdm: Uses tqdm progressbar
            plot: Plot learning curves with mathplotlib
        THe model must have vectorized environments wrapped by Monitor.
    """

    def __init__(self,update_interval=10_000,kind='tqdm'):
        super().__init__()
        self.update_interval=update_interval
        self.kind=kind
    
    def _on_training_start(self):
        self.total_timesteps=self.locals['total_timesteps']
        if self.kind=='tqdm':
            self.progress_bar = tqdm(total=self.total_timesteps)
        elif self.kind=='plot':
            pass
    
    def _on_step(self):    
        if self.num_timesteps % self.update_interval ==0:
            self.update_progress()
        return True

    def _on_training_end(self):
        self.update_progress()
        if self.kind=='tqdm':
            self.progress_bar.close()
            self.progress_bar = None        

    def update_progress(self):
        dft=self.model.history['train']
        if dft.shape[0]>0:
            dft=dft.groupby(((dft['CumSteps']-1)//self.update_interval+1)*self.update_interval).agg(['min','mean','max','sum'])
            dfe=self.model.history['eval']
            dfe=dfe.groupby(((dfe['CumSteps']-1)//self.update_interval+1)*self.update_interval).agg(['min','mean','max'])
            if self.kind=='tqdm':
                self.progress_bar.update(self.num_timesteps-self.progress_bar.n)
                self.progress_bar.set_postfix_str("Train reward:{:6.3f},Episode length:{:6.0f}".format(dft[('Rewards','mean')].iloc[-1],dft.iloc[-1][('Lengths','mean')]))
                if dfe.shape[0]>0:
                    self.progress_bar.set_description_str("Eval reward:{:6.3f}".format(dfe[('Rewards','mean')].iloc[-1]))
            elif self.kind=='plot':
                clear_output(wait=True)
                fig,ax=plt.subplots(1,1,figsize=(12,4))
                #ax.set_xlim((0,self.total_timesteps))
                ax.plot(dft.index,dft[('Rewards','mean')],'b',label='Train mean:{:6.3f}'.format(dft[('Rewards','mean')].iloc[-1]))
                ax.plot(dft.index,dft[('Rewards','min')],'g.',label='Train min:{:6.3f}'.format(dft[('Rewards','min')].iloc[-1]))
                ax.plot(dft.index,dft[('Rewards','max')],'g.',label='Train max:{:6.3f}'.format(dft[('Rewards','max')].iloc[-1]))
                ax.plot([],'b',label='Episode length:{:4.0f}'.format(dft.iloc[-1][('Lengths','mean')]))
                ax.plot([],'b',label='It/sec:{:4.0f}'.format(dft[('Lengths','sum')].sum()/dft[('Times','max')].max()))
                if dfe.shape[0]>0:
                    ax.plot(dfe.index,dfe[('Rewards','mean')],'r',label='Eval mean:{:6.3f}'.format(dfe[('Rewards','mean')].iloc[-1]))
                ax.legend()
                plt.show()
