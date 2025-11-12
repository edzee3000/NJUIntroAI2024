package controllers.depthfirst;

import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.Random;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import ontology.Types;
import ontology.Types.WINNER;
import tools.ElapsedCpuTimer;


public class Agent extends AbstractPlayer {
    //一些类模板的参数
    protected Random randomGenerator;
    protected ArrayList<Observation> grid[][];
    protected int block_size;

    ArrayList<StateObservation> pastState = new ArrayList<StateObservation>();      //存储已经走过的状态序列用来判断是否陷入重复死循环
    ArrayList<Types.ACTIONS> Actions = new ArrayList<Types.ACTIONS>();           //存储走过的动作序列
    int count = 0;                                                        //count记录下需要返回的路径下标
    boolean have_path = false;                                            //是否已搜索到路径
    //初始化构造函数
    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer)
    {
        randomGenerator = new Random();
        grid = so.getObservationGrid();
        block_size = so.getBlockSize();
    }

    //判断当前状态之前是否到达过
    boolean test_past_state(StateObservation state){                  
        for (int i=0; i<pastState.size(); i++) if(state.equalPosition( pastState.get(i) ))return true;  //注意动态数组不能够想普通数组一样[i]  需要使用get方法 
        return false;
    }

    //核心递归代码！！！
    boolean get_DFS_Actions(StateObservation stateObs, ElapsedCpuTimer elapsedTimer){
        pastState.add(stateObs);  //将当前状态加入已走过的状态
        ArrayList<Types.ACTIONS> all_available_actions = stateObs.getAvailableActions();
        for(int i=0;i<all_available_actions.size();i++){                                             
            //尝试当前局面所有可以的动作
            StateObservation stCopy = stateObs.copy();                          //新建一个当前状态的副本，用于模拟施加动作
            stCopy.advance(all_available_actions.get(i));                                             //施加动作
            Actions.add(all_available_actions.get(i));                                                //将当前动作加入已走过的动作
            
            if(stCopy.getGameWinner() == WINNER.PLAYER_WINS)  return true;     //如果获胜则返回true表示已找到
            if(test_past_state(stCopy) || stCopy.isGameOver()) Actions.remove(Actions.size() - 1);  //如果动作施加后的状态之前已经到达过或者游戏失败就尝试下一个动作，并且需要删除当前已经加进去的这个动作
            else{                                                               //动作施加后的是一个新的状态
                if(get_DFS_Actions(stCopy,elapsedTimer))  return true;               //递归进行深度优先搜索   
                else  Actions.remove(Actions.size() - 1);                  //当前动作递归搜索失败，尝试下一个动作
            }
        }
        pastState.remove(pastState.size() - 1);                          //当前局面没有办法成功，删除当前局面
        return false;
    }
    //给出行动的代码
    public Types.ACTIONS act(StateObservation stateObs, ElapsedCpuTimer elapsedTimer) {

        ArrayList<Observation>[] npcPositions = stateObs.getNPCPositions();
        ArrayList<Observation>[] fixedPositions = stateObs.getImmovablePositions();
        ArrayList<Observation>[] movingPositions = stateObs.getMovablePositions();
        ArrayList<Observation>[] resourcesPositions = stateObs.getResourcesPositions();
        ArrayList<Observation>[] portalPositions = stateObs.getPortalsPositions();
        grid = stateObs.getObservationGrid();

        // printDebug(npcPositions,"npc");
        // printDebug(fixedPositions,"fix");
        // printDebug(movingPositions,"mov");
        // printDebug(resourcesPositions,"资源位置为");
        // printDebug(portalPositions,"传送门位置为");
        // System.out.println();               
        if(!have_path){
            have_path = get_DFS_Actions(stateObs, elapsedTimer);           //如果没有搜索过则进行搜索，搜索成功后将flag置为true，并且已将路径存储在Actions中
        }
        if(count < Actions.size()){
            Types.ACTIONS action=Actions.get(count);
            System.out.println("第"+count+"步执行的动作为："+action);  
            try{Thread.sleep(400); } // 休眠0.4秒让整体更加直观
            catch (InterruptedException e) {
                // 如果线程在休眠期间被中断，会抛出InterruptedException异常
                System.err.println("Thread was interrupted.");
                // 可以选择重新设置中断标志
                Thread.currentThread().interrupt();}
            count++;
            return action;                                  //搜索到路径后将Actions中的动作依次执行
        }
        return null;
    }

    /**
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

    /**
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