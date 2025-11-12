package controllers.Astar;

import core.game.Observation;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import java.awt.Graphics2D;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import ontology.Types;
import ontology.Types.WINNER;
import tools.ElapsedCpuTimer;
import tools.Vector2d;

public class Agent extends AbstractPlayer{
    //一些类模板的参数
    protected Random randomGenerator;
    protected ArrayList<Observation> grid[][];
    protected static int block_size;


    ArrayList<StateObservation> StateSequence = new ArrayList<StateObservation>();      //存储已经走过的状态序列用来判断是否陷入重复死循环
    //ArrayList<Types.ACTIONS> Actions = new ArrayList<Types.ACTIONS>();           //存储走过的动作序列
    int count = 0;                                                        //count记录下需要返回的路径下标
    boolean have_path = false;                                            //是否已搜索到路径
    ArrayList<Node> open_list=new ArrayList<Node>(); //存储候选节点
    ArrayList<Node> close_list=new ArrayList<Node>(); //存储已经遍历过的节点
    ArrayList<Types.ACTIONS> Actions = new ArrayList<Types.ACTIONS>();           //存储要走的动作序列


    protected static Vector2d goalpos;                                                                //目标的位置
    protected static Vector2d keypos;                                                                 //钥匙的位置
    //protected static double goal_keyDistance;                              =new ArrayList<Node>();                            //钥匙与目标之间的曼哈顿距离
    protected int flag=0;



    //初始化构造函数
    public Agent(StateObservation so, ElapsedCpuTimer elapsedTimer)
    {
        randomGenerator = new Random();
        grid = so.getObservationGrid();
        block_size = so.getBlockSize();
    }


     //判断当前状态之前是否到达过
     boolean test_in_openlist_and_closelist(StateObservation state){   
                       
        for(Node open_node :open_list) if(state.equalPosition(open_node.state)) return true; 
        for(Node close_node :close_list) if(state.equalPosition(close_node.state)) return true; 
        return false;
    }

    


    //A*搜索核心代码
    boolean AstarSearch(StateObservation stateObs, ElapsedCpuTimer elapsedTimer)
    {
        StateObservation init_state=stateObs;
        Node root_node = new Node(init_state,null,null);
        open_list.add(root_node);
        // close_list.add(root_node);//将根节点加入到close_list当中去
        // for(Types.ACTIONS action : stateObs.getAvailableActions())
        // {StateObservation stCopy = stateObs.copy();stCopy.advance(action);Node newNode=new Node(stCopy,root_node,action);open_list.add(newNode); }//初始化将可允许的添加进openlist（注意openlist里面是节点）
        // print_open_close();
        //从openlist中选出F最小的节点加入到closelist中然后删除，产生新的getAvailableActions对应的状态节点并加入openlist中
        //重复上述操作直到游戏失败or游戏胜利or到了死胡同
        while(!open_list.isEmpty())
        {
            //在openlist中寻找F最小的node并删除，加入到closelist中
            Node min_node=search_min_node();    open_list.remove(min_node);     close_list.add(min_node);
            System.out.println("最小节点状态位置为："+min_node.state.getAvatarPosition());
            // try{Thread.sleep(1000); } // 放在这里用来中断调试
            // catch (InterruptedException e) {// 如果线程在休眠期间被中断，会抛出InterruptedException异常
            //     System.err.println("Thread was interrupted.");
            //     // 可以选择重新设置中断标志
            //     Thread.currentThread().interrupt();}


            //更新openlist
            for(Types.ACTIONS action :min_node.state.getAvailableActions())
            {StateObservation stCopy = min_node.state.copy();  stCopy.advance(action);  Node newNode=new Node(stCopy,min_node,action); 
            //如果找到了终点的话则停止并产生更新全局Arraylist的动作序列（利用链表的思想）
            if(stCopy.getGameWinner() == WINNER.PLAYER_WINS)//如果游戏胜利的话
            {   System.out.println("游戏胜利");
                Node parent=newNode; 
                // System.out.println("最后一步动作为:"+parent.act);
                while(parent!=null && parent.act!=null){System.err.println("每一步动作为："+parent.act);Actions.add(parent.act);    parent=parent.last;}
                Collections.reverse(Actions);
                return true;
            }
            if (flag==0 && !newNode.has_key)flag=0; else if(flag==0&&newNode.has_key){System.out.println("拿到钥匙");flag=1;    open_list=new ArrayList<Node>();   open_list.add(newNode);  close_list=new ArrayList<Node>(); break;}//如果拿到钥匙了那就清空openlist和closelist
            
            if(stCopy.isGameOver()){System.out.println("游戏失败");//try{Thread.sleep(3000);} // 放在这里用来中断调试
            // catch (InterruptedException e) {// 如果线程在休眠期间被中断，会抛出InterruptedException异常
            //     System.err.println("Thread was interrupted.");
            //     // 可以选择重新设置中断标志
            //     Thread.currentThread().interrupt();}
                close_list.add(newNode);}
            

            if(test_in_openlist_and_closelist(stCopy)){//System.out.println("在里面");
            continue;}
            if(!test_in_openlist_and_closelist(stCopy)){//System.out.println("不在里面");
            open_list.add(newNode);}
            
            }

        }

        System.out.println("游戏失败");
        try{Thread.sleep(1000); } // 放在这里用来中断调试
            catch (InterruptedException e) {// 如果线程在休眠期间被中断，会抛出InterruptedException异常
                System.err.println("Thread was interrupted.");
                // 可以选择重新设置中断标志
                Thread.currentThread().interrupt();}
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
            //初始化目标位置，钥匙位置，以及钥匙与目标之间的曼哈顿距离
            goalpos = stateObs.getImmovablePositions()[1].get(0).position;
            keypos = stateObs.getMovablePositions()[0].get(0).position;
            //goal_keyDistance = Math.abs(goalpos.x - keypos.x) + Math.abs(goalpos.y - keypos.y);
            System.out.println("目标位置为："+goalpos+"\t钥匙位置为："+keypos); 

            have_path = AstarSearch(stateObs, elapsedTimer);           //如果没有搜索过则进行搜索，搜索成功后将flag置为true，并且已将路径存储在Actions中
            System.out.println("是否有路径："+have_path);
            if(!have_path)return null;
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






    //定义一个Node节点类最终通过last进行回溯
    public static class Node{
        public StateObservation state;
        public Node last=null;
        public int cost;
        public boolean has_key=false;
        public Vector2d PlayerPosition;
        public Types.ACTIONS act=null; 
        public int level=0;//level表示已经有多少cost了
        public Node(StateObservation state,Node last_node,Types.ACTIONS act){
            this.state=state;
            this.PlayerPosition = state.getAvatarPosition();              //精灵的位置
            if (last_node!=null)
            {this.last=last_node;
            this.level=last_node.level+1;
            this.act=act;
            }
            if(this.state.getAvatarPosition().equals(keypos) || (this.last!=null&&this.last.has_key))   
                this.has_key=true;
            this.cost=this.F();
        }
        //定义一个计算cost的函数F(x)=G(x)+H(x)，其中G为已经花费的cost，H为预计还需要花费的cost
        //不妨令预计的花费为人到钥匙的距离+钥匙到终点距离（如果人没有拿到钥匙）；人到终点的距离（如果人拿到了钥匙）
        public int F()
        {
            int G=this.level;//G表示已经花费的cost代价
            int H;
            int distance_player_key= (int)(Math.abs(PlayerPosition.x - keypos.x) + Math.abs(PlayerPosition.y - keypos.y))/block_size;
            int distance_player_goal=(int)(Math.abs(PlayerPosition.x - goalpos.x) + Math.abs(PlayerPosition.y - goalpos.y))/block_size;
            int distance_key_goal=(int)(Math.abs(keypos.x - goalpos.x) + Math.abs(keypos.y - goalpos.y))/block_size;
            if (this.has_key) H=distance_player_goal;//如果已经有钥匙了
            else H=distance_key_goal+distance_player_key;
            return G+H;
        }
    }

    public void print_open_close()
    {
        for(Node node:open_list)
            System.out.println("openlist里面有的node的位置有："+node.PlayerPosition);
        for(Node node:close_list)
            System.out.println("closelist里面有的node的位置有："+node.PlayerPosition);
    }
   


    //在openlist中寻找F最小的节点
    Node search_min_node()
    {
        Node min_Node=null;
        for(Node node:open_list)
        {
            if(min_Node==null)min_Node=node;
            if(min_Node.cost>=node.cost)min_Node=node;
        }
        return min_Node;
    }


    //调试函数打印一些结果（也不是非常重要可以不用看）
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

    //画图函数（并不是非常重要说实话）
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












