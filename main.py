import csv 
import numpy as np
import random
import matplotlib.pyplot as plt

from collections import defaultdict

from pymoo.indicators.hv import HV
import copy 
import os

# import sys
# sys.setrecursionlimit(1000) # might need to increase (maybe? debug code left if needed)

# DIR = "resources/" # + "test-example-n4.txt" # loading data

def load_data(file_name):
    """
        Loads the data from the file and returns the nodes and problem_dict
    """
    weights = []
    values = []

    with open(file_name,'r') as f:
        reader = csv.reader(f, delimiter = '|')
        rows = list(reader)

        # Left this way for readability
        PROBLEM_NAME = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        KNAPSACK_DATA_TYPE = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        DIMENSION = int(rows[0][0].split(":")[1].strip())
        rows.pop(0)
        NUMBER_OF_ITEMS = int(rows[0][0].split(":")[1].strip())
        rows.pop(0)
        CAPACITY = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        SPEED_MIN = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        SPEED_MAX = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        RENTING_RATIO = rows[0][0].split(":")[1].strip()
        rows.pop(0)
        EDGE_WEIGHT_TYPE = rows[0][0].split(":")[1].strip()
        rows.pop(0)

        problem_dict = {
            "PROBLEM_NAME": PROBLEM_NAME,
            "KNAPSACK_DATA_TYPE": KNAPSACK_DATA_TYPE,
            "DIMENSION": int(DIMENSION),
            "NUMBER_OF_ITEMS": int(NUMBER_OF_ITEMS),
            "CAPACITY": float(CAPACITY),
            "SPEED_MIN": float(SPEED_MIN),
            "SPEED_MAX": float(SPEED_MAX),
            "RENTING_RATIO": float(RENTING_RATIO),
            "EDGE_WEIGHT_TYPE": EDGE_WEIGHT_TYPE
        }

        rows.pop(0) # NODE_COORD_SECTION	(INDEX, X, Y): 
        
        nodes = []

        while rows[0][0].split()[0] != "ITEMS":
            content = []
            _content = rows.pop(0)[0].split(" ")

            for i in range(len(_content)):
                if _content[i] == "": continue
                sub_content = _content[i].split("\t")
                for j in range(len(sub_content)):
                    if sub_content[j] == "": continue
                    else: content.append(sub_content[j])    

            node = [int(content[0]), float(content[1]), float(content[2]), []]
            nodes.append(node)

        rows.pop(0) # ITEMS SECTION	(INDEX, PROFIT, WEIGHT, ASSIGNED NODE NUMBER):

        while rows != []:
            content = []
            _content = rows.pop(0)[0].split(" ")

            for i in range(len(_content)):
                if _content[i] == "": continue
                sub_content = _content[i].split("\t")
                for j in range(len(sub_content)):
                    if sub_content[j] == "": continue
                    else: content.append(sub_content[j])

            node_id = int(content[3])
            node_content = [int(content[1]), int(content[2]), int(content[0])] # profit, weight, bag_id

            nodes[node_id-1][3].append(node_content)
            
        return nodes, problem_dict

def pareto_front(solutions, indices):
    """
    Finds the Pareto front from a list of solutions.
    """
    pareto_front = []
    pareto_front_ids = []
    
    for i, sol_i in enumerate(solutions):
        is_dominated = False
        for j, sol_j in enumerate(solutions):
            if np.all(np.array(sol_j) <= np.array(sol_i)) and np.any(np.array(sol_j) < np.array(sol_i)):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(sol_i)
            pareto_front_ids.append(indices[i])
    
    return pareto_front, pareto_front_ids

def get_consecutive_pareto_fronts(solutions):

    fronts = []
    front_indices = []

    remaining_solutions = solutions[:]
    remaining_front_indices = list(range(len(solutions)))

    while remaining_solutions != []:
        front, front_indcy = pareto_front(remaining_solutions, remaining_front_indices)
        remaining_solutions = [sublist for sublist in remaining_solutions if sublist not in front]
        remaining_front_indices = [sublist for sublist in remaining_front_indices if sublist not in front_indcy]
        fronts.append(front)
        front_indices.append(front_indcy)
    
    return fronts, front_indices

class Utils:
    """
        Utils class

        Distance Calculation
        Profit Calculation
        Weight Calculation

        and other utility functions such as: can_visit(), can_take() etc.
    """

    problem_dict = {}
    nodes = []

    def __init__(self, nodes, problem_dict):
        self.nodes = nodes
        self.problem_dict = problem_dict
        pass

    def __has_duplicates__(self, individual):
        """
            Checks if the list has duplicates
        """
        if len(individual) <= 1:
            # Too short to have duplicates
            return False
        
        # collect the node ids
        list = [x[0] for x in individual]

        if len(list) == len(set(list)):
            return False
        else:
            return True
    
    def get_weight(self, route):
        """
            Calculates the weight of the route

            Input: List of lists
                [[node_id, [bool, bool, ...]], ...]

            Output: int
        """

        weight = 0.0
       
        for node in route:
            ws = node[1]

            # if ws is type(int):
            #     print(ws)
            #     print("\n")
            #     print(node[1])
            #     raise("INT?")

            # if sum(ws) == 0: continue
            if len(self.nodes[node[0]-1][3]) == 0: continue

            for i in range(len(ws)):
                weight += ws[i] * self.nodes[node[0]-1][3][i][1]

        return weight
    
    def get_max_weight(self):
        """
            returns max weight / capacity 
        """
        return self.problem_dict["CAPACITY"]

    def get_distance(self, node1, node2):
        """
            Calculates the distance between two nodes

            Input: List of lists
                [node_id, x, y, [profit, weight, bag_id], [profit, weight, bag_id], ...]
        """
        return ((node1[1] - node2[1])**2 + (node1[2] - node2[2])**2)**0.5
    
    def get_profit(self, route):
        """
            Calculates the profit of the route

            Input: List of lists
                [[node_id, [bool, bool, ...]], ...]

            Output: int
        """

        profit = 0.0

        for node in route:
            vs = node[1]

            if len(self.nodes[node[0]-1][3]) == 0: continue

            for i in range(len(vs)):
                profit += vs[i] * self.nodes[node[0]-1][3][i][0]

        return profit
    

    def is_valid_individual(self, individual):
        """
            Checks if the individual is valid
        """
        if self.__has_duplicates__(individual):
            return 2
        
        if self.get_weight(individual) > self.problem_dict["CAPACITY"]:
            return 1
        
        return 0
    
    def __calc_velocity__(self, weight):
        """
            Calculates the velocity of the individual
        """

        if weight <= 0:
            return self.problem_dict["SPEED_MAX"]

        W_MAX = self.problem_dict["SPEED_MAX"]
        W_MIN = self.problem_dict["SPEED_MIN"]

        velocity = W_MAX - (weight / self.problem_dict["CAPACITY"]) * (W_MAX - W_MIN)

        if velocity < W_MIN or weight >= self.problem_dict["CAPACITY"]:
            velocity = W_MIN
        elif velocity >= 1.0:
            velocity = W_MAX

        return velocity
    
    def get_instances_of_repeated_gene(self, individual):
        """
            Finds the instances of all repeated genes (not including the first instance)
        """

        indices_dict = defaultdict(list)

        lst = [x[0] for x in individual]

        for idx, value in enumerate(lst):
            indices_dict[value].append(idx)
        

        return [indices for indices in indices_dict.values() if len(indices) > 1]

    def fitness_calc_time(self, individual):
        """
            Calculates the fitness of the individual
        """

        # add start and end point 
        individual = [self.nodes[0]] + individual[:] + [self.nodes[0]]
        
        history = []
        time = 0
        weight = 0.0
        velocity = self.problem_dict["SPEED_MAX"]

        # picks up first item at time 0 with max speed 
        history.append(individual[0])
        weight += self.get_weight([individual[0]])

        for i in range(1, len(individual)):
            distance = self.get_distance(self.nodes[individual[i-1][0]-1], self.nodes[individual[i][0]-1])
            velocity = self.__calc_velocity__(weight)
            time += distance / velocity
            weight += self.get_weight([individual[i]])
        return time
    
    def get_max_locations(self):
        """
            Returns the max number of locations a individual can visit 
        """
        return self.problem_dict["DIMENSION"]-1 # 0 is our fist number so 280 -> 279 for index / len()    

class GA:
    """
        Genetic Algorithm class
        for GECO2019
    """

    # Generation Report Counters
    __GENERATION_INDEX__ = 0 

    # Util functions 
    UTIL = None

    # Number of mutations (based on dyn * pop_size)
    DYNAMIC_MUTATION = 50
    DYNAMIC_CROSSOVER = 50

    pop = []
    pop_size = 100
    num_nodes = 0

    def __init__(self, nodes, problem_dict, pop_size = 250, dyn_mutation = 0.5, dyn_crossover = 0.5):
        """
            Initializes the GA with the nodes and problem_dict

            Input:
                nodes: List of lists
                    [[node_id, x, y, [profit, weight, bag_id], [profit, weight, bag_id], ...], ...]
                problem_dict: Dictionary
                    {
                        ...
                        "DIMENSION": int,              
                        "NUMBER_OF_ITEMS": int,
                        "CAPACITY": int,
                        "SPEED_MIN": float,
                        "SPEED_MAX": float,
                        "RENTING_RATIO": float,
                    }
        """
        self.__GENERATION_INDEX__ = 0 
        
        self.pop_size = pop_size
        self.num_nodes = len(nodes)

        self.DYNAMIC_MUTATION = int(pop_size*dyn_mutation)
        self.DYNAMIC_CROSSOVER = int(pop_size*dyn_crossover)

        self.UTIL = Utils(nodes, problem_dict)

        # generate initial population
        print("\n Generating Initial Population")
        for i in range(pop_size):
            individual = self.generate_individual()
            self.pop.append(individual)
            # print(i)
        print("============================")

    def get_solutions(self):
        """
            returns best solutions so far
        """
        return self.pop

    def generate_gene(self, gene_id=-1, full_zeros=False):
        """
            generates a single gene

            location 1 is always the first and last so never allowed to be generated
        """
        if full_zeros: 
            return [gene_id, self.mutation_single_node_full(gene_id, full_zeros=full_zeros)]

        gene_id = random.randint(1, self.num_nodes-1)
        return [gene_id, self.mutation_single_node_full(gene_id)]

    def generate_individual(self, retries = 3):
        """
            Generates a new individual with the given nodes

            Input: retries (int)
                Number of times to retry generating a new individual
        """
        # random number between 0 and num_nodes

        individual_id = [] #used purely to avoid duplicates and reduced complexity
        duplicate_retry = 0 # flag counter for duplicate retries

        individual = []

        incomplete = True
        while incomplete:
            # generate node/gene
            r = random.randint(1, self.num_nodes-1)

            if r not in individual_id:
                individual_id.append(r)
                # generate bags

                individual.append([r, self.mutation_single_node_full(r)])
                # TODO: remove [0] values (too lazy tonight todo it but easy fix)
            # else: retries -= 1
            if retries <= 0:
                break
                    # [individual_id[-1], individual_bags[-1]]
            individual_validity = self.UTIL.is_valid_individual(individual)

            if individual_validity == 0: # valid node & bags
                duplicate_retry = 0 # reset duplicate retry 
            elif individual_validity == 1: # invalid node due to weight 
                individual.pop()
                retries -= 1 
            elif individual_validity == 2: # invalid node due to duplicates
                # invalid node due to duplicates
                if duplicate_retry > 15: # duplicates can occur due to chance so try again
                    # too many duplicates found
                    break
                duplicate_retry += 1
                individual.pop()

            # if full individual mutation must take over 
            if len(individual) >= self.UTIL.get_max_locations()-1:
                incomplete = False


        return self.fix_individual_validity(individual) # final checks (should be clear but this is the init pop)

    def mutation_single_node_full(self, individual_node_id, full_zeros=False):
        """
            Fully mutates a single node's selected bags in the individual

            Input: node_id (int)
                The node to mutate
        """

        bag_length = len(self.UTIL.nodes[individual_node_id][3])

        if bag_length == 1:
            return [random.choice([0, 1])]
        
        if bag_length == 0:
            raise("NO")
            return [0]
        
        if full_zeros:
            zeros = [0 for _ in range(bag_length)]
            return zeros
        
        # random chance 0 or 1 for each bag (50% chance each)
        bags = [random.choice([0, 1]) for _ in range(bag_length)]

        # one bag must be selected else impossible to visit
        # if sum(bags) == 0: # TODO: remove all instances of this check as its incorrect 
        #     bags[random.randint(0, bag_length-1)] = 1

        return bags
    
    def correct_missing(self, individual):
        """
            gets all the missing IDs for an individual 
        """

        if len(individual) >= self.UTIL.get_max_locations():
            return individual

        # collect the node ids
        list = [x[0] for x in individual]
        list = sorted(list)

        for num in range(1, self.UTIL.get_max_locations()+1):
            if num not in list:
                # individual = self.mutation_new_gene(individual=individual, gene_id=num, full_zeros=True)
                individual = [self.generate_gene(gene_id=num, full_zeros=True)] + individual[:]
            
        return individual
        

    
    def mutation_new_gene(self, individual, full_zeros=False, gene_id=-1):
        """
            Mutation adds a new gene into a position in the individual 

            Note: checks need to be done after to check if this is a valid individual
        """
        max_mutation_flag = True # if false maxed out

        # split location 
        r = random.randint(0, len(individual))
        gene = []

        if len(individual) >= self.UTIL.get_max_locations()-1 and (not full_zeros):
            # CAN ONLY REGENERATE A GENE NOT REPLACE IT SO
            # SAVE GENE ID , MUTATE BAGS BY MUTATION , MOVE ON
            max_mutation_flag = False 
            r1 = random.randint(0, len(individual)-1)
            gene = individual[r1]
            gene[1] = [1 - bit if random.random() < 0.5 else bit for bit in gene[1]]
            individual.pop(r1)
        else:
            if full_zeros == True:
                gene = self.generate_gene(gene_id=gene_id, full_zeros=full_zeros)
            else: gene = self.generate_gene()
        
        # first attempt 
        new = individual[:r] + [gene] + individual[r:]

        # loop till a gene is found that can fit 
        while self.UTIL.__has_duplicates__(new) == True and max_mutation_flag:
            new = individual[:r] + [self.generate_gene()] + individual[r:]      

        # new = self.fix_individual_validity(new)

        return new

    def fix_individual_validity(self, individual, fill_knapsack=True):
        """
            Fixes an Individual (harsh - full checks)
        """

        flag_weight = 0
        flag_repeated = 0

        # get repeated gene ids
        gene_ids = self.UTIL.get_instances_of_repeated_gene(individual)

        # if too large of an individual to have unique genes
        if len(individual) > self.UTIL.get_max_locations():
            # drop repeated first
            while len(gene_ids) > 0:
                # random gene_ids lst 
                r_x = random.randint(0, len(gene_ids)-1) # select set  
                r_y = random.randint(0, len(gene_ids[r_x])-1) # select id 
                individual.pop(gene_ids[r_x][r_y])
                
                # re-calc list 
                gene_ids = self.UTIL.get_instances_of_repeated_gene(individual)

                # if problem resolved go back to other checks 
                if len(individual) > self.UTIL.get_max_locations():
                    break
            
            # if still occurs 
            while len(individual) > self.UTIL.get_max_locations():
                # drop random genes 
                r = random.randint(0, len(individual)-1)
                individual.pop(r)
        
        # recalculate to ensure that all repeated genes are removed
        gene_ids = self.UTIL.get_instances_of_repeated_gene(individual)

        # repeated gene dropping (random from remaining) [list of list of repeated locations [[1, 2] [3, 4, 5]]
        while len(gene_ids) > 0:
            flag_repeated = 1 
            # random gene_ids lst 
            r_x = random.randint(0, len(gene_ids)-1) # select set  
            r_y = random.randint(0, len(gene_ids[r_x])-1) # select id 
            individual.pop(gene_ids[r_x][r_y])
            
            # re-calc list 
            gene_ids = self.UTIL.get_instances_of_repeated_gene(individual)


        # weight checking / removal of excess 
        while self.UTIL.get_weight(individual) > self.UTIL.get_max_weight():
            flag_weight = 1
            # too heavy so need to drop a random gene
            individual.pop(random.randint(0, len(individual)-1))


        # if not overweight and no repeated flag hit this is completed 
        if flag_weight == 1 and flag_repeated == 0: # weights is fixed already and no repeated 
            return individual
        else:
            # repopulate with genes to add more to the bag
            new = individual 

            # if still under weight try to add more 
            while self.UTIL.get_weight(new) < self.UTIL.get_max_weight() and fill_knapsack:
                # replace 
                individual = new 

                # not role to mutate here only fix so this is acceptable so far if max locations
                if len(new) >= self.UTIL.get_max_locations() - 1:
                    break

                # find new gene 
                new = self.mutation_new_gene(new)
        
        # individual MUST visit all cities 
        if len(individual) != self.UTIL.get_max_locations():
            individual = self.correct_missing(individual)

        return individual

    def crossover(self, parent1, parent2):
        """
            Crossover where produced child is one side parent1 and the other parent2
        
            Note: parent1 and parent2 should be selected randomly as it will always take the first half of parent1 and the second half of parent2 
            then splice them together. Plus error checking to ensure it is
        """

        # choose a random point to crossover
        r = random.randint(0, len(parent1)-1)

        # create the child
        child = parent1[:r] + parent2[r:]

        child = self.fix_individual_validity(child)
        child = self.correct_missing(child)

        return child

    def mutation_drop(self, individual):
        """
            drop random elements to go below weight threshold 
        """

        if len(individual) <= 1:
            print(len(individual) <= 2)
            return individual

        child = individual.pop(random.randint(0, len(individual)-1))
        return child
    
    def select_random(self, front, num_to_select):
        """
            Selects random elements up till num_to_select
        """

        if len(front) < num_to_select:
            raise "ERROR: invalid front provided: must comply with len(front) < num_to_select "
        
        selected_front = []

        while len(selected_front) < num_to_select:
            r = random.randint(0, len(front)-1)
            selected_front.append(front.pop(r))

        return selected_front

    def select_s_metric(self, front, front_ids, num_to_select):
        """
            takes a front and returns a list of selected ids in order
        """

        if len(front_ids) < num_to_select:
            raise "ERROR: invalid front provided: must comply with len(front) < num_to_select "
        
        if len(front_ids) <= 1: # already has determined behaviour  
            return self.select_random(front_ids, num_to_select)
        
        front_np = np.array(front)

        # epsilon value (max value for prioritizing edge solutions) 
        # default was 1 but is now replaced
        epsilon_x = (sum(front_np[:, 0]))/len(front)
        epsilon_y =  (sum(front_np[:, 1]))/len(front)
        
        # reference point 
        reference_point = np.array([max(front_np[:, 0]) + epsilon_x, max(front_np[:, 1]) + epsilon_y])  # Reference point is chosen slightly worse than the worst solution in the front

        # calculate reference hypervolume and init
        hv = HV(reference_point)
        front_hypervolume = hv.do(front_np)

        # Generate contribution values 
        hv_contrib = []
        for i in range(len(front)):
            reduced_front = front
            reduced_front = np.array(reduced_front)
            reduced_front =  np.delete(reduced_front, i, axis=0)
            reduced_front_hypervolume = hv.do(reduced_front)

            # calculate contribution 
            reduced_contrib = front_hypervolume - reduced_front_hypervolume
            hv_contrib.append(reduced_contrib)

        # order list 
        ids = [index for index, score in sorted(zip(front_ids, hv_contrib), key=lambda x: x[1], reverse=True)]

        return ids[:num_to_select]

    def selection(self, pop, pop_size):
        """
            Survival Function 
        """

        # get fitness 
        x, y = self.gen_fitness(pop)
        xy = [[xi, yi] for xi, yi in zip(x, y)]
        # Get consecutive Pareto fronts
        front, front_ids = get_consecutive_pareto_fronts(xy)

        solutions = []

        for i in range(len(front_ids)):
            if len(solutions) + len(front_ids[i]) <= pop_size:
                # no checks needed as all are best solutions found 
                for fr in front_ids[i]:
                    solutions.append(fr)
            elif ((len(solutions) + len(front_ids[i])) >= pop_size) and (len(solutions) < pop_size):
                # Select from final front 
                _front_ids = self.select_s_metric(front[i], front_ids[i], (pop_size - len(solutions))) 
                for fr in _front_ids:
                    solutions.append(fr)
            else:   
                # Complete 
                break

        return solutions
                
    def generation(self):
        """
            Perform a single Generation

            generate new population   
        """

        print(f"Generation {self.__GENERATION_INDEX__}")
        self.__GENERATION_INDEX__ += 1 

        num_genes_mutation = 1  # genes to attempt to mutate 

        child_pop = []

        def check_dupe(c, popul=child_pop):
            """
                Check for duplicates within child_pop 
            """
            for indv in popul:
                dupe = True
                # same length 
                if len(indv) == len(c):

                    for i in range(len(indv)-1):
                        # if same position 
                        if indv[i][0] == c[i][0]:
                            # if same bags selected 
                            if indv[i][1] is not c[i][1]:
                                dupe = False

                        else:
                            dupe = False
                        
                        # break out 
                        if dupe == False:
                            break
                else:
                    dupe = False

                # Stop searching individual dupe found 
                if dupe == True:
                    return dupe
                                
            return False
        
        dupes_counter = 0

        # add parents to child pop (checks are mostly unnecessary but are still a layer of security)
        for parent in self.pop:
            # append only if non-dupe (mostly for initial pop)
            # if self.__GENERATION_INDEX__ <= 1:
            #     child_pop.extend(self.pop)
            #     break

            if not check_dupe(parent):
                child_pop.append(copy.deepcopy(parent))

        # replace pop with child pop to remove dupes from causing damage 
        print(f"Parent Population {len(child_pop)} / {len(self.pop)}")
        self.pop = child_pop
            

        # mutation_replace_gene
        dupes_counter = 0
        for i in range(self.DYNAMIC_MUTATION):
            r = random.randint(0, len(self.pop)-1)
            child = self.pop[r]

            for j in range(num_genes_mutation):
                # child = self.mutation_replace_gene(child) # TODO bugfix
                child = self.mutation_new_gene(child)
                child = self.fix_individual_validity(child) # more checks for validity 
                child = self.correct_missing(child) # will exit instantly if correct size
            
            # child_pop.append(child)
            if not check_dupe(child):
                child_pop.append(child)
            else:
                dupes_counter += 1

        print(f"Mutation Dupes Found {dupes_counter}")
        dupes_counter = 0


        # crossover
        for i in range(self.DYNAMIC_CROSSOVER):
            r1 = random.randint(0, len(self.pop)-1)
            r2 = random.randint(0,  len(self.pop)-1)

            if r1 == r2:
                i -= 1
                continue

            p1 = self.pop[r1]
            p2 = self.pop[r2]


            # crossover
            child = self.crossover(p1, p2)
            child = self.correct_missing(child)

            if not check_dupe(child):
                child_pop.append(child)
            else:
                dupes_counter += 1

        print(f"Crossover Dupes Found {dupes_counter}")
        dupes_counter = 0

        
        # selection 
        new_pop = []

        # fix integrity of solutions that get through the gaps (TODO: find where 277/278 length leak is)
        for i in range(len(child_pop)):
            child_pop[i] = self.correct_missing(child_pop[i])

        # select next generation 
        new_pop_ids = self.selection(child_pop, self.pop_size)

        for id in new_pop_ids:
            # print(len(child_pop[id]))
            new_pop.append(child_pop[id])

        self.pop = new_pop

        # Generation Report (END)
        print(f"""Unique Population {len(child_pop)} / {self.pop_size + 
                                                  self.DYNAMIC_CROSSOVER +
                                                  self.DYNAMIC_MUTATION} :: {len(new_pop)}""")
        print(f"Pareto Length {len(set(new_pop_ids))} / {len(new_pop_ids)}")
        print(f"======================")

    def gen_fitness(self, pop=None):
        """
            Generates the fitness of the population
        """

        x = []
        y = []

        if pop is None:
            pop = self.pop

        for individual in pop:
            y.append(-self.UTIL.get_profit(individual))
            x.append(self.UTIL.fitness_calc_time(individual))
        
        return x, y

# ==========================================================================
#   VIS
# ==========================================================================

def display(ga, title="Consecutive Pareto Fronts"):
    """
        Visualisation Function 
    """

    x, y = ga.gen_fitness()
    x_label = "time"
    y_label = "-profit"

    solutions = [[xi, yi] for xi, yi in zip(x, y)]

    # Get consecutive Pareto fronts
    pareto_fronts, _ = get_consecutive_pareto_fronts(solutions)

    # Plot all the solutions
    solutions_np = np.array(solutions)
    print(f"Total Solutions in Plot {len(solutions_np)}")

    # colour map with distinct colours
    cmap = plt.get_cmap('tab10', len(pareto_fronts)) 

    # Plot each Pareto front with a line connecting the points 
    for i, front in enumerate(pareto_fronts):
        front_np = np.array(sorted(front, key=lambda x: x[0]))
        color = cmap(i)  # Get the color for the i-th front
        plt.scatter(front_np[:, 0], front_np[:, 1], label=f'Front {i+1} / len {len(front_np)}', color=color)
        plt.plot(front_np[:, 0], front_np[:, 1], color=color, linestyle='-', marker='o')

    # Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Show legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()

def Backup_copy(ga, title="genG9task", folder_name="report/backup", gen_num = 0, copy_location="results_backup", n = 100):
    """
        Saves a copy of both the GA and Vis in unique folders (for first run*) 
    """

    report(ga=ga, task_name=title, folder_name=folder_name, n=n)

    x, y = ga.gen_fitness()
    x_label = "time"
    y_label = "-profit"

    solutions = [[xi, yi] for xi, yi in zip(x, y)]

    # Get consecutive Pareto fronts
    pareto_fronts, _ = get_consecutive_pareto_fronts(solutions)

    # Plot all the solutions
    solutions_np = np.array(solutions)
    print(f"Total Solutions in Plot {len(solutions_np)}")

    # colour map with distinct colours
    cmap = plt.get_cmap('tab10', len(pareto_fronts)) 

    # Plot each Pareto front with a line connecting the points 
    for i, front in enumerate(pareto_fronts):
        front_np = np.array(sorted(front, key=lambda x: x[0]))
        color = cmap(i)  # Get the color for the i-th front
        plt.scatter(front_np[:, 0], front_np[:, 1], label=f'Front {i+1} / len {len(front_np)}', color=color)
        plt.plot(front_np[:, 0], front_np[:, 1], color=color, linestyle='-', marker='o')

    # Labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Show legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.savefig(
        f"{folder_name}/{title}.png",
        dpi=300,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0.1,
    )

    plt.clf()

def report(ga, folder_name="results/_bin", task_name="bin", n=100):
    """
        Generates Report .x .f files
            saves them in given location 
    """

    # Create Report folder structure 
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created!")

    all_solutions = ga.get_solutions()
    solution_ids = ga.selection(all_solutions, n)
    solutions = []

    for id in solution_ids:
        solutions.append(all_solutions[id])

        ################################# .x ################


    page = ""

    for i, solution in enumerate(solutions):
        # Define the filename dynamically, e.g., solution1.x, solution2.x, etc.
        file_path = os.path.join(folder_name, f"{task_name}.x")
        
        packing_plan = ""
        path_taken = "1"

        # build path
        for j in range(len(solution)):
            path_taken = f"{path_taken} {solution[j][0]}"

        # build packing plan 
        for j in range(len(solution)):
            pack = ""

            for bool in solution[j][1]:
                if pack == "":
                    pack = f"{bool}"
                else:
                    pack = f"{pack} {bool}"

            if packing_plan == "":
                packing_plan = pack
            else:
                packing_plan = f"{packing_plan} {pack}"

        if page == "":
            page = f"{path_taken}\n{packing_plan}\n"
        page = f"{page}\n{path_taken}\n{packing_plan}\n"


    # Write the solution to the file
    with open(file_path, "w") as file:
        file.write(f"{page}")

    ################################# .f ################

    print(f"Solution {i} written to {file_path}\{task_name}.x")
    
    # new directory 
    file_path = os.path.join(folder_name, f"{task_name}.f")

    # generate time / profit 
    x, y = ga.gen_fitness()
    for y_i in range(len(y)):
        y[y_i] = -y[y_i]

    time_profit = ""
    for i in range(len(x)):
        if time_profit == "":
            time_profit = f"{x[i]} {y[i]}\n"
        else:
            time_profit = f"{time_profit}{x[i]} {y[i]}\n"

    with open(file_path, "w") as file:
        file.write(f"{time_profit}")

# ==========================================================================
#   MAIN
# ==========================================================================

def run(problem="a280-n1395", num_gen=1000, save_point=5, backup_point=50, n=100):
    """
        Runs the problem

        n is the 
    """

    nodes, problem_dict = load_data(f"resources/{problem}.txt")

    ga = GA(nodes, problem_dict, pop_size=100, dyn_crossover=2, dyn_mutation=2)

    # MAIN LOOP
    for i in range(num_gen-1):
        ga.generation()

        if i % save_point == 0:
            # save copy of latest report 
            report(ga, "results", f"G9-{problem}", n)
            # display(ga) # only show if you don't want to save

        if i % backup_point == 0:
            Backup_copy(ga=ga, title=f"{ga.__GENERATION_INDEX__}-G9-{problem}", folder_name="report/backup", n=n)


    # final generation 
    ga.pop_size = n         # set population size (this will change results after next gen)
    ga.generation()         # run gen and will scale result to pop_size 

    report(ga, "results", f"G9-{problem}")

run(problem="a280-n279", num_gen=10000, n=100)
# run(problem="a280-n1395", num_gen=10000, n=100)
# run(problem="a280-n2790", num_gen=10000, n=100)
