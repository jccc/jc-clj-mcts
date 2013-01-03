(ns jc-clj-mcts.core)

;;MCTS in clojure 
;;http://randomcomputation.blogspot.com/2013/01/monte-carlo-tree-search-in-clojure.html


;;;================================TTT

(def wins [[0 1 2] [3 4 5] [6 7 8] ;cols 
           [0 3 6] [1 4 7] [2 5 8] ;rows
           [0 4 8] [2 4 6]])       ;diags


(defn opp-player [p] (if (= p :X) :O :X))

(defn blank? [x] (number? x))


(defn win-check [state player]
  (loop [wins wins]
    (if (empty? wins) false
      (let [[a b c] (first wins)]
        (if (and (= player (get state a))
                 (= player (get state b))
                 (= player (get state c)))
           true
           (recur (rest wins)))))))


(defn ttt-terminal? [state]
  (if (win-check state :X)         {:draws 0 :x-win 1 :o-win 0}
    (if (win-check state :O)       {:draws 0 :x-win 0 :o-win 1} 
      (if (not-any? blank? state)  {:draws 1 :x-win 0 :o-win 0} false))))
           
  


(defn ttt-gen-children [state to-move]
  (for [[i v] (zipmap (range) state)
        :when (blank? v)]
    (assoc state i to-move)))


(defn ttt-playout
  "Returns value of playout simulation for tic tac toe"
  [state to-move]  
  (loop [state state to-move to-move]  
    (if-let [result (ttt-terminal? state)] result
      (recur (rand-nth (ttt-gen-children state to-move))             
             (opp-player to-move)))))

;;;================================UCT

(def init-record 
  "The data associated with a board-state in the mem"
  {:visits 0  :draws    0 
   :x-win  0  :o-win    0 
   :chldn  [] :to-move :X})

(def init-mem
  "Creates initial mem for TTT"
  (let [init-state [ 0 0 0 
                     0 0 0 
                     0 0 0]]
    {init-state 
     (assoc init-record :chldn (ttt-gen-children init-state :X))}))
  

(defn uct-value 
  "Value of a state based on gathered statistics. Currently not 
   actually 'uct' value (see paper)."
  [{:keys [visits x-win o-win draws to-move]}]
  (case (opp-player to-move) 
    :X (/ (+ x-win (/ draws 2))
          visits)
    :O (/ (+ o-win (/ draws 2))
          visits)
    :default 0))


(defn uct-sample
  "The random sampling function for a board state."
  [state mem func times]
  (loop [result {:draws 0 :x-win 0 :o-win 0} times times]
    (if (< times 1) result      
      (recur (reduce 
               (fn [m [k v]]
                 (update-in m [k] + v))
               result
               (func state (get-in mem [state :to-move])))
             (dec times)))))


(defn uct-select
  "Selects highest rated child of state"
  [mem state]
  (let [chldn (get-in mem [state :chldn])      
        explrd (remove 
                 #(zero? (get-in mem [% :visits] 0)) 
                 chldn)]
    (if (empty? explrd)
      (rand-nth chldn)
      (apply max-key #(uct-value (get mem %)) explrd))))

        

(defn uct-unexplored [mem state]
  "Unexplored children of state"
  (for [c (get-in mem [state :chldn]
                  (ttt-gen-children state (get-in mem [state :to-move])))
        :when (zero? (get-in mem [c :visits] 0))] c))


(defn uct-backprop 
  "Backpropagates child value to the parent"
  [mem path {:keys [x-win o-win draws] :as stats}] 
  (if (empty? path) mem
    (recur
      (-> mem
        (update-in [(first path) :x-win] + x-win)
        (update-in [(first path) :o-win] + o-win)
        (update-in [(first path) :draws] + draws)
        (update-in [(first path) :visits] inc))
      (rest path)
      stats)))


(defn- add-child 
  "Helper to creates child-record for the mem."
  [mem parent-state child-state]  
  (let [to-move (get-in mem [parent-state :to-move])
        child-record (get mem 
                          child-state
                          (assoc init-record
                                 :chldn (ttt-gen-children 
                                          child-state 
                                          (opp-player to-move))
                                 :to-move (opp-player to-move)))]
    (assoc mem child-state child-record)))
    
  

(defn uct-grow 
  "Estimates a child's value and adds it to the tree."
  [mem path]  
  (let [leaf (first path)
        chld (rand-nth (uct-unexplored mem leaf))
        valu (uct-sample chld mem ttt-playout 1)]    
    (-> mem
      (add-child leaf chld)
      (uct-backprop (cons chld path) valu))))


(defn learn-iteration [mem state]
  "The core algorithm; a single analysis of a state. Searches the tree
   for an unexplored child, estimates the child's value, adds
   it to the tree, and backpropagates the value up the path."
  (loop [mem mem, state state, path (list state)]
    (if-let [result (ttt-terminal? state)]
      (uct-backprop mem path result)
      (if (not-empty (uct-unexplored mem state))
        (uct-grow mem path)
        (let [ch (uct-select mem state)]
          (recur mem ch (cons ch path)))))))


(defn learn-state [mem state budget]
  "Analyzes a board state using the UCT algorithm. Iterates
   learn-iteration until budget is exhausted."
  (loop [mem mem budget budget]    
    (if (< budget 1) mem
      (recur (learn-iteration mem state) (dec budget)))))



;;;================================RUN

(defn print-board [board]
  "Pretty print a board state"
  (println 
    (apply format "%s %s %s \n%s %s %s \n%s %s %s \n"
           (map #(case % :X "X" :O "O" 0 "_") board))))



(defn play-game 
  "Retains memory built from analyses of past moves"
  [[mem _]]  
  (let [uctp (rand-nth [:X :O])]
    (loop [mem mem
           board-state [0 0 0 0 0 0 0 0 0] 
           to-move :X]
      (if-let [{:keys [draws x-win o-win]} (ttt-terminal? board-state)] 
        [mem
         (hash-map 
          :uct (if (= uctp :X) x-win o-win) 
          :rnd (if (= uctp :X) o-win x-win) 
          :draws draws)]
        (if (= uctp to-move)          
          (let [mem (learn-state mem board-state 30)
                move (uct-select mem board-state)]
            (recur mem move (opp-player to-move)))
          (let [move (rand-nth (get-in mem [board-state :chldn]))
                mem (if (contains? mem move) mem
                      (assoc mem move 
                             (assoc init-record
                                    :chldn (ttt-gen-children 
                                             move 
                                             (opp-player to-move))
                                    :to-move (opp-player to-move))))]
            (recur mem move (opp-player to-move))))))))



(defn play-game-no-mem
  "Does not retain memory over moves to allow for 
   effectiveness assessment based on computational budget"
  [budget]
  (let [uctp (rand-nth [:X :O])]
    (loop [board-state [0 0 0 0 0 0 0 0 0] 
           to-move :X]
;      (print-board board-state)
      (if-let [{:keys [draws x-win o-win]} (ttt-terminal? board-state)] 
         (hash-map 
          :uct (if (= uctp :X) x-win o-win) 
          :rnd (if (= uctp :X) o-win x-win) 
          :draws draws)
        (if (= uctp to-move)          
          (let [mem (learn-state (hash-map
                                   board-state                                   
                                   (assoc init-record
                                          :to-move to-move
                                          :chldn (ttt-gen-children 
                                                   board-state 
                                                   to-move)))
                                 board-state 
                                 budget)
                move (uct-select mem board-state)]
            (recur move (opp-player to-move)))
          (let [move (rand-nth (ttt-gen-children board-state to-move))]                
            (recur move (opp-player to-move))))))))


(defn- update-stats
  "Helper function"
  [curr new]
  (reduce
    (fn [m [k v]] (update-in m [k] + v))
    curr new))



(defn uct-v-rand
  "Plays n games of uct vs rand retaining the analysis memory
   across games"
  [n]  
  (loop [mem init-mem
         games 0
         stats {:uct 0 :rnd 0 :draws 0}]
    (if (> games n) [mem stats]
      (let [[mem result] (play-game mem)]
        (recur mem 
               (inc games)
               (update-stats stats result))))))

 
;;;This script generates the results table in the blog post     
;(let [data (for [b [0 1 2 3 4 5 10 100]] ;computational budgets
;             (list b (take 50 (repeatedly #(play-game-no-mem b)))))
;      stats (map
;              (fn [[b d]]
;                (let [avgs {:uct   (float (/ (reduce + (map :uct d)) (count d)))
;                            :rnd   (float (/ (reduce + (map :rnd d)) (count d)))
;                            :draws (float (/ (reduce + (map :draws d)) (count d)))}]
;                  (list b avgs)))
;              data)]
;  (pprint stats))