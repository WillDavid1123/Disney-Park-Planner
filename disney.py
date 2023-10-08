# disney.py
# Plan a trip to Walt Disney World's Magic Kingdom theme park using a genetic algorithm.

from graphics import *
import csv
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import time
import pdb

ROOT = os.path.dirname(os.path.abspath(__file__))
# Files decided on in main()
# FILE_IMG = 'hollywoodstudios_map.png'
# FILE_LOC = 'ridelocations-hs.txt'
# FILE_RIDE = 'ridetimes-hs.txt'
# FILE_WAIT = 'waittimes-hs.txt'
# FILE_FUN = 'ridefun-hs.txt'
# PARK = "Hollywood Studios"
COLOR_RIDE = '#3e50c4'
COLOR_PATH = '#3e50c4'
PAUSE = 0.02  # increase to show planned route with longer time delay
CIRCSIZE = 8 # Size of circles on map


def main():
    # Check which park the user wants to use
    # Note: for Epcot and Hollywood Studios, unpack the 'Extra Parks' zipped folder 
    #   and add the files to the same folder as disney.py
    while True:
        print("Which park would you like to visit?")
        print(" 1) Magic Kingdom")
        print(" 2) Epcot")
        print(" 3) Hollywood Studios")
        park = input()
        if park == '1':
            PARK = "Magic Kingdom"
            FILE_IMG = 'magickingdom.png'
            FILE_LOC = 'ridelocations.txt'
            FILE_RIDE = 'ridetimes.txt'
            FILE_WAIT = 'waittimes.txt'
            FILE_FUN = 'ridefun.txt'
            break
        elif park == '2':
            PARK = "Epcot"
            FILE_IMG = 'epcot_map.png'
            FILE_LOC = 'ridelocations-e.txt'
            FILE_RIDE = 'ridetimes-e.txt'
            FILE_WAIT = 'waittimes-e.txt'
            FILE_FUN = 'ridefun-e.txt'
            break
        elif park == '3':
            PARK = "Hollywood Studios"
            FILE_IMG = 'hollywoodstudios_map.png'
            FILE_LOC = 'ridelocations-hs.txt'
            FILE_RIDE = 'ridetimes-hs.txt'
            FILE_WAIT = 'waittimes-hs.txt'
            FILE_FUN = 'ridefun-hs.txt'
            break
        else:
            print("Please pick one of the numbers above\n")

    # Variable Setup
    max_time = 600 # Number of minutes for fitness

    # Load data from file
    locations = readlocations(os.path.join(ROOT, FILE_LOC))
    time = readridetimes(os.path.join(ROOT, FILE_RIDE))
    wait = readwaittimes(os.path.join(ROOT, FILE_WAIT))
    fun = readfun(os.path.join(ROOT, FILE_FUN))

    gui = makegui(os.path.join(ROOT, FILE_IMG), PARK)
    addrides(gui, locations, CIRCSIZE)

    # Run genetic algorithm
    bestplan = GA(time, wait, fun, np.array([x for x in fun.keys()]), locations, max_time)
    rides = list(fun.keys())
    # bestplan = np.random.permutation(rides)
    addroute(gui, locations, bestplan)

    while True:
        key = gui.checkKey()
        if key == 'Escape':
            break
        elif key == 'd':
            pdb.set_trace()
        elif key == 'a':
            addroute(gui, locations, bestplan)
        elif key == 'c':
            clearroute(gui)

        # pt = gui.checkMouse()
        # if pt:
        #     print(pt)


def addpath(gui, a, b):
    '''Add a path from point a to point b on the map.'''
    line = Line(Point(a[0], a[1]), Point(b[0], b[1]))
    line.setWidth(4)
    line.setOutline(COLOR_PATH)
    line.draw(gui)


def addrides(gui, locations, size):
    '''Add markers for relevant rides on the gui using ride locations.'''
    for ride, xy in locations.items():
        # print(ride, xy)
        circ = Circle(Point(xy[0], xy[1]), size)
        circ.setOutline(COLOR_RIDE)
        circ.setWidth(4)
        circ.draw(gui)


def addroute(gui, locations, plan):
    '''Display the planned route on the map, where the plan is a list of the rides to visit
    sequentially. The first and last location in the plan must be 'Entrance'; if it is not,
    the code adds it to the plan automatically (i.e. the GA doesn't need to include that).'''
    plan = np.array(plan)  # just in case
    if plan[0] != 'Entrance':
        plan = np.insert(plan, 0, 'Entrance')
    if plan[-1] != 'Entrance':
        plan = np.append(plan, 'Entrance')

    print("Here's the plan...")
    for i in range(plan.shape[0] - 1):
        print(i, plan[i])
        addpath(gui, locations[plan[i]], locations[plan[i + 1]])
        time.sleep(PAUSE)
    print()


def clearroute(gui):
    '''Undraw any routes on the map.'''
    for obj in gui.items[18:]:
        obj.undraw()


def makegui(imgfile, park):
    '''Helper function to initialize relevant graphics.'''
    # Read image from file
    img = Image(Point(0, 0), imgfile)
    hei = img.getHeight()
    wid = img.getWidth()
    img.move(wid / 2, hei / 2)

    # Create graphics window
    gui = GraphWin(f"Walt Disney World -- { park }", wid, hei)
    img.draw(gui)

    return gui


def readfun(txtfile):
    '''Helper function to load the fun factor (in whatever scale you have chosen!) of each ride on the map. Output is a dictionary where keys are rides and values indicate the level of fun.'''
    with open(txtfile) as f:
        rides = list(csv.reader(f))
    fun = {ride[0]: float(ride[1]) for ride in rides}

    return fun


def readlocations(txtfile):
    '''Helper function to load the location (in pixels) of each ride on the map. Output is a dictionary where keys are rides and values are lists of [x, y] location on the image.'''
    with open(txtfile) as f:
        rides = list(csv.reader(f))
    locations = {ride[0]: [int(ride[1]), int(ride[2])] for ride in rides}

    return locations


def readridetimes(txtfile):
    '''Helper function to load the ride time (in minutes) of each ride on the map. Output is a dictionary where keys are rides and values are times.'''
    with open(txtfile) as f:
        rides = list(csv.reader(f))
    times = {ride[0]: float(ride[1]) for ride in rides}

    return times


def readwaittimes(txtfile):
    '''Helper function to load the ride time (in minutes) of each ride on the map. Output is a dictionary where keys are rides and values are times.'''
    with open(txtfile) as f:
        rides = list(csv.reader(f))
    times = {ride[0]: float(ride[1]) for ride in rides}

    return times


def GA(ride_time, wait, fun, rides, locations, max_time):
    '''Main hub for GA functions'''
    # Setup Variables (Changeable)
    pop_size = 200
    num_offspring = 200
    tourn_size = 10 #Size of tournament for parent selection
    pc = 0.7
    pm = 0.15
    walk_times = walking(rides, locations)
    num_genes = 30
    max_iter = 10000
    improv_min = 500 #Terminate if no improvements to fitness have been made in X generations
    reduce_fun = 3 #Amount of fun™ lost for a ride after each time it is ridden

    # Initialization (Don't touch)
    population = np.random.randint(0, rides.shape[0], size=(pop_size, num_genes))
    score = fitness(population, rides, walk_times, wait, ride_time, fun, max_time, reduce_fun)
    order = np.argsort(score)[::-1]
    population = population[order, :]
    current_best_plan = population[0]
    current_best_score = max(score)
    no_improvement = 0
    print(f"Generation 0: Starting Best Fitness = {current_best_score}")

    for i in range(max_iter):
        #Select parents for next gen
        parents = parent_select(population, num_offspring, tourn_size)
        
        #Create children using those parents
        children = crossover(parents, pc)
        children = mutation(children, rides, pm)
        
        #Prepare the next generation with survivor selection
        population, score = select_survivors(population, children, rides, walk_times, wait, ride_time, fun, max_time, reduce_fun)

        #Print whenever the new highest score is higher than the current best
        if score[0] > current_best_score:
            current_best_score = score[0]
            current_best_plan = population[0]
            print(f"Generation {i + 1}: Improved Fitness = {score[0]}")
            no_improvement = 0
        else:
            no_improvement += 1

        #Terminate if there is no improvement to score in X generations 
        if no_improvement == improv_min:
            print(f"Generation {i + 1}: Algorithm Terminated - No Improvement in {improv_min} Generations")
            break
    print()

    #Cut down best plan to only include rides actually ridden while converting the plan from integers to ride names
    best_plan_string, best_plan_int = cut_down(current_best_plan, rides, walk_times, wait, ride_time, max_time)

    #Print the amount of fun™ and the time it takes to execute the best plan
    plan_time = findTime(best_plan_int, rides, walk_times, wait, ride_time, max_time)
    print(f"With this plan, you'll have {current_best_score} fun™ in {math.floor(plan_time / 60):.0f} hours and {(int)(plan_time % 60.0):.0f} minutes")

    #Return the best plan found
    return best_plan_string


def walking(rides, locations):
    '''Create 2d array of the walk times between each ride'''
    locals = np.append('Entrance', rides) #Add entrance to the list of rides for the first walk of every plan
    walk_times = np.zeros((rides.shape[0] + 1, rides.shape[0] + 1))
    #Find distances between all locations
    for i, ride1 in enumerate(locals):
        x1 = locations[ride1][0]
        y1 = locations[ride1][1]
        for j, ride2 in enumerate(locals):
            x2 = locations[ride2][0]
            y2 = locations[ride2][1]
            walk_times[i, j] = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
    #Convest distances to walking times (walk time = 0.015 * distance)
    walk_times = walk_times * 0.015
    return walk_times


def fitness(population, rides, walk_times, wait, ride_time, fun, max_time, fun_lost):
    '''Fitness function for population (Disney plans)'''
    ind_fun = np.zeros(population.shape[0])
    for i, individual in enumerate(population):
        curr_time = 0
        prev_ride = 0 #Entrance
        plan_fun = fun.copy() #Fun™ of each ride for the given plan
        for gene in individual:
            # pdb.set_trace()
            aGene = gene + 1 #Adjusted gene for addition of 'Entrance' in walk_times
            ride_name = rides[gene]
            curr_time += walk_times[prev_ride][aGene]
            curr_time += wait[ride_name]
            curr_time += ride_time[ride_name]
            prev_ride = aGene
            if curr_time > max_time:
                break
            ind_fun[i] += plan_fun[ride_name]
            plan_fun[ride_name] -= fun_lost #Reduce fun™ of ride for each time it's ridden
    return ind_fun


def parent_select(population, num_offspring, tourn_size):
    '''Tournament selection (size k) of n parents from a given population.'''
    players = np.random.randint(population.shape[0], size=(num_offspring, tourn_size))
    winners = players.min(axis=1)
    return population[winners, :]


def crossover(parents, pc):
    '''Crossover parents using Uniform Crossover'''
     # Extract pairs of parents
    parent1 = parents[::2, :]
    parent2 = parents[1::2, :]

    # Determine crossover masks
    mask = np.random.rand(*parent1.shape) < 0.5
    nocrossover = np.random.rand(parent1.shape[0]) > pc
    mask[nocrossover, :] = False

    # Apply uniform crossover
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]

    children = np.r_[child1, child2]  # merge children

    return children

def mutation(children, rides, pm):
    '''Perform random resetting on children with probability pm.'''
    newvalues = np.random.randint(0, rides.shape[0], size=children.shape)
    mask = np.random.rand(*children.shape) <= pm
    children[mask] = newvalues[mask]

    return children

def select_survivors(population, mutants, rides, walk_times, wait, ride_time, fun, max_time, reduce_fun):
    '''Fitness-based replacement (and sorting) of population + mutants.'''
    candidates = np.vstack((population, mutants))
    score = fitness(candidates, rides, walk_times, wait, ride_time, fun, max_time, reduce_fun)
    order = np.argsort(score)[::-1]  # sort fitness in ascendin order (we want to minimize!)
    candidates = candidates[order, :]  # rearrange population based on fitness
    score = score[order]

    survivors = candidates[:population.shape[0], :]
    score = score[:population.shape[0]]
    return survivors, score

def cut_down(plan, rides, walk_times, wait, ride_time, max_time):
    '''Cut down given plan to only include rides actually ridden'''
    curr_time = 0
    prev_ride = 0 #Entrance
    actual_plan = []
    for ride in plan:
        aGene = ride + 1 #Adjusted gene for addition of 'Entrance' in walk_times
        ride_name = rides[ride]
        curr_time += walk_times[prev_ride][aGene]
        curr_time += wait[ride_name]
        curr_time += ride_time[ride_name]
        prev_ride = aGene
        if curr_time > max_time:
            break
        actual_plan.append(rides[ride])
    # pdb.set_trace()
    return actual_plan, plan[:len(actual_plan)]

def findTime(plan, rides, walk_times, wait, ride_time, max_time):
    '''Calculate and return the amount of time a given plan would take'''
    plan_time = 0
    prev_ride = 0 #Entrance
    for gene in plan:
        aGene = gene + 1 #Adjusted gene for addition of 'Entrance' in walk_times
        ride_name = rides[gene]
        plan_time += walk_times[prev_ride][aGene]
        plan_time += wait[ride_name]
        plan_time += ride_time[ride_name]
        prev_ride = aGene
        if plan_time > max_time:
            break
    return plan_time

if __name__ == '__main__':
    main()
