import csv 
import numpy as np
import random


DIR = "resources/" # + "test-example-n4.txt" # loading data

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
            "CAPACITY": int(CAPACITY),
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

        weight = 0

        for node in route:
            ws = node[1]
            for i in range(len(ws)):
                weight += ws[i] * self.nodes[node[0]-1][i]

        return weight

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

        profit = 0

        for node in route:
            profit += node[3][0][0]

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

class GA:
    """
        Genetic Algorithm class
        for GECO2019
    """

    UTIL = None

    pop = []
    pop_size = 100
    num_nodes = 0

    def __init__(self, nodes, problem_dict, pop_size = 100):
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
        self.pop_size = pop_size
        self.num_nodes = len(nodes)

        self.UTIL = Utils(nodes, problem_dict)

        # generate initial population
        for i in range(pop_size):
            individual = self.generate_individual()
            self.pop.append(individual)

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
            # generate node
            r = random.randint(0, self.num_nodes-1)

            if r not in individual_id:
                individual_id.append(r)
                # generate bags

                individual.append([r, self.mutation_single_node_full(r)])
                # TODO: remove [0] values (too lazy tonight todo it but easy fix)
            # else: retries -= 1
            if retries <= 0:
                # print("Length: ", len(individual_id))
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
                    # print(f"Could not max out the individual capacity len: {len(individual)}")
                    break
                duplicate_retry += 1
                individual.pop()
                
        return individual

    def mutation_single_node_full(self, individual_node_id):
        """
            Fully mutates a single node's selected bags in the individual

            Input: node_id (int)
                The node to mutate
        """

        bag_length = len(self.UTIL.nodes[individual_node_id][3])

        if bag_length == 1:
            return [1]
        
        if bag_length == 0:
            # print("[0] No bags available for node " + str(individual_node_id) + "!")
            return [0]
        
        # TODO: CHECK THIS FOR MORE THAN 1 BAG

        # random chance 0 or 1 for each bag (50% chance each)
        bags = [random.choice([0, 1]) for _ in range(bag_length)]

        # one bag must be selected else impossible to visit
        if sum(bags) == 0: 
            bags[random.randint(0, bag_length-1)] = 1

        return bags

    def mutation_bags(self):
        """
            Mutates the bags of the individuals 

            swaps the bags around to see if the solution improves 
                DOES NOT CHANGE THE VISITED NODES
        """
        pass

    def selection(self):
        pass

nodes, problem_dict = load_data(DIR + "a280-n279.txt")

ga = GA(nodes, problem_dict)