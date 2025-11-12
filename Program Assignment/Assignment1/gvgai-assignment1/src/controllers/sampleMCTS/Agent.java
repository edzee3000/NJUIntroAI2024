package controllers.sampleMCTS;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import java.util.ArrayList;
import java.util.Random;
import ontology.Types;
import tools.ElapsedCpuTimer;

/**
 * Created with IntelliJ IDEA.
 * User: ssamot
 * Date: 14/11/13
 * Time: 21:45
 * This is a Java port from Tom Schaul's VGDL - https://github.com/schaul/py-vgdl
 */
public class Agent extends AbstractPlayer {

    public static int NUM_ACTIONS;
    public static int ROLLOUT_DEPTH = 10;
    public static double K = Math.sqrt(2);
    public static Types.ACTIONS[] actions;

    /**
     * Random generator for the agent.智能体的随机生成器。
     */
    private SingleMCTSPlayer mctsPlayer;

    /** 
     * 这个函数是构造函数
     * Public constructor with state observation and time due.具有状态观察和到期时间的公共构造函数
     * @param so state observation of the current game.当前游戏状态观察
     * @param elapsedTimer Timer for the controller creation.用于创建控制器的计时器
     */
    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer)
    {
        //Get the actions in a static array.获取静态数组中的操作
        ArrayList<Types.ACTIONS> act = so.getAvailableActions();
        actions = new Types.ACTIONS[act.size()];
        for(int i = 0; i < actions.length; ++i)
        {
            actions[i] = act.get(i);
        }
        NUM_ACTIONS = actions.length;

        //Create the player.创建玩家
        mctsPlayer = new SingleMCTSPlayer(new Random());
    }


    /**
     * Picks an action. This function is called every game step to request an
     * action from the player.
     * 选择一个动作。每个游戏步骤都会调用此函数来请求
     * 玩家的动作
     * @param stateObs Observation of the current state.观察当前状态
     * @param elapsedTimer Timer when the action returned is due.
     * @return An action for the current state
     */
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) 
    {

        ArrayList<Observation> obs[] = stateObs.getFromAvatarSpritesPositions();
        ArrayList<Observation> grid[][] = stateObs.getObservationGrid();

        //Set the state observation object as the new root of the tree.
        mctsPlayer.init(stateObs);

        //Determine the action using MCTS...
        int action = mctsPlayer.run(elapsedTimer);

        //... and return it.
        return actions[action];
    }
 
    /**
     * Function called when the game is over. This method must finish before CompetitionParameters.TEAR_DOWN_TIME,
     *  or the agent will be DISQUALIFIED
     * 游戏结束时调用的函数。该方法必须在CompetitionParameters.TEAR_DOWN_TIME之前完成，
     * 否则智能体将被取消资格
     * @param stateObservation the game state at the end of the game 游戏结束时的游戏状态
     * @param elapsedCpuTimer timer when this method is meant to finish. 该方法何时结束的计时器
     */
    public void result(StateObservation stateObservation, ElapsedCpuTimer elapsedCpuTimer)
    {
        //Include your code here to know how it all ended.在此处包含您的代码以了解这一切是如何结束的
        //System.out.println("Game over? " + stateObservation.isGameOver());
    }


}
