package controllers.sampleRandom;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import java.awt.Graphics2D;
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

    /**
     * Random generator for the agent.用于生成随机数的 Random 对象。
     */
    protected Random randomGenerator;

    /**
     * Observation grid.存储游戏观察网格的二维数组，每个元素包含该位置的所有 Observation 对象。
     */
    protected ArrayList<Observation> grid[][];

    /**
     * block size每个游戏块的像素尺寸
     */
    protected int block_size;


    /**
     * Public constructor with state observation and time due.构造函数接收当前游戏状态的观察和计时器，用于初始化智能体。
     * @param so state observation of the current game.
     * @param elapsedTimer Timer for the controller creation.
     */
    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer)
    {
        randomGenerator = new Random();
        grid = so.getObservationGrid();
        block_size = so.getBlockSize();
    }


    /**
     * Picks an action. This function is called every game step to request an
     * action from the player.这是智能体的核心方法，它在每个游戏步骤被调用，以请求玩家的动作。
     * @param stateObs Observation of the current state.
     * @param elapsedTimer Timer when the action returned is due.
     * @return An action for the current state
     */
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {

        ArrayList<Observation>[] npcPositions = stateObs.getNPCPositions();//获取NPC位置
        ArrayList<Observation>[] fixedPositions = stateObs.getImmovablePositions();//获取固定位置（不可移动）
        ArrayList<Observation>[] movingPositions = stateObs.getMovablePositions();//获取可移动位置
        ArrayList<Observation>[] resourcesPositions = stateObs.getResourcesPositions();//获取资源位置
        ArrayList<Observation>[] portalPositions = stateObs.getPortalsPositions();//获取传送门位置
        grid = stateObs.getObservationGrid();

        /*printDebug(npcPositions,"npc");
        printDebug(fixedPositions,"fix");
        printDebug(movingPositions,"mov");    用于调试目的，打印每种类型精灵的数量和观察结果。
        printDebug(resourcesPositions,"res");
        printDebug(portalPositions,"por");
        System.out.println();               */

        Types.ACTIONS action = null;//初始化一个动作变量，这个变量将存储智能体选择的动作。
        StateObservation stCopy = stateObs.copy();//创建当前游戏状态的一个副本，以便在尝试不同的动作时不会影响原始状态。

        double avgTimeTaken = 0;//初始化用于计算平均时间的变量。
        double acumTimeTaken = 0;//
        long remaining = elapsedTimer.remainingTimeMillis();// 获取剩余的时间（毫秒），这是智能体必须在其中做出决策的时间。
        int numIters = 0;// 初始化迭代计数器

        int remainingLimit = 5;//设置一个剩余时间的最小限制，以确保智能体有足够的时间做出决策。
        while(remaining > 2*avgTimeTaken && remaining > remainingLimit)//只要剩余时间大于两倍的平均时间或大于限制时间，就继续迭代
        {
            ElapsedCpuTimer elapsedTimerIteration = new ElapsedCpuTimer();//创建一个新的计时器实例，用于测量每次迭代的时间。
            ArrayList<Types.ACTIONS> actions = stateObs.getAvailableActions();//获取当前状态下可用的所有动作。
            int index = randomGenerator.nextInt(actions.size());//随机选择一个动作索引。
            action = actions.get(index);//根据随机索引选择一个动作

            stCopy.advance(action);//使用选择的动作模拟游戏状态的前进
            if(stCopy.isGameOver())//查模拟后的状态是否是游戏结束状态，如果是，则重置状态副本。
            {
                stCopy = stateObs.copy();
            }

            numIters++;//增加迭代计数器。
            acumTimeTaken += (elapsedTimerIteration.elapsedMillis()) ;// 更新累积时间
            //System.out.println(elapsedTimerIteration.elapsedMillis() + " --> " + acumTimeTaken + " (" + remaining + ")");
            avgTimeTaken  = acumTimeTaken/numIters;//计算平均时间
            remaining = elapsedTimer.remainingTimeMillis();//更新剩余时间
        }

        return action;//返回选择的动作
    }

    /**用于调试目的，打印每种类型精灵的数量和观察结果。
     * Prints the number of different types of sprites available in the "positions" array.
     * Between brackets, the number of observations of each type.
     * @param positions array with observations.
     * @param str identifier to print
     */
    private void printDebug(ArrayList<Observation>[] positions, String str)
    {
        if(positions != null){
            System.out.print(str + ":" + positions.length + "(");
            for (int i = 0; i < positions.length; i++) {
                System.out.print(positions[i].size() + ",");
            }
            System.out.print("); ");
        }else System.out.print(str + ": 0; ");
    }
    
    /**允许智能体在游戏屏幕上绘制信息，用于调试。它在每个网格单元格中绘制观察到的精灵的类别。
     * Gets the player the control to draw something on the screen.
     * It can be used for debug purposes.
     * @param g Graphics device to draw to.
     */
    public void draw(Graphics2D g)
    {
        int half_block = (int) (block_size*0.5);
        for(int j = 0; j < grid[0].length; ++j)
        {
            for(int i = 0; i < grid.length; ++i)
            {
                if(grid[i][j].size() > 0)
                {
                    Observation firstObs = grid[i][j].get(0); //grid[i][j].size()-1
                    //Three interesting options:
                    int print = firstObs.category; //firstObs.itype; //firstObs.obsID;
                    g.drawString(print + "", i*block_size+half_block,j*block_size+half_block);
                }
            }
        }
    }
}
