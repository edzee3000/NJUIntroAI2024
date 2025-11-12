package controllers.doNothing;

import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import ontology.Types.ACTIONS;
import tools.ElapsedCpuTimer;

public class Agent extends AbstractPlayer{


	/**
	 * initialize all variables for the agent 初始化代理的所有变量
	 * @param stateObs Observation of the current state. 当前状态的观察
     * @param elapsedTimer Timer when the action returned is due. 返回的操作到期的计时器。
	 */
	public Agent(StateObservation stateObs, ElapsedCpuTimer elapsedTimer){
	}
	
	/**
	 * return ACTION_NIL on every call to simulate doNothing player 每次调用时都返回 ACTION_NIL 以模拟 doNothing 玩家
	 * @param stateObs Observation of the current state. 当前状态的观察
     * @param elapsedTimer Timer when the action returned is due.
	 * @return 	ACTION_NIL all the time
	 */
	@Override
	public ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {
		return Types.ACTIONS.ACTION_NIL;
	}
}
