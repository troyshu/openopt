#!/usr/bin/python
import numpy as np
import random
random.seed(15)
import sys

#from PIL import Image, ImageDraw, ImageFont
from math import sqrt

def rand_seq(size):
    '''generates values in random order
    equivalent to using shuffle in random,
    without generating all values at once'''
    values=np.arange(size).tolist()
    for i in range(size):
        # pick a random index into remaining values
        j=i+int(random.random()*(size-i))
        # swap the values
        values[j],values[i]=values[i],values[j]
        # return the swapped value
        yield values[i] 

def all_pairs(size):
    '''generates all i,j pairs for i,j from 0-size'''
    for i in rand_seq(size):
        for j in rand_seq(size):
            yield (i,j)

def reversed_sections(tour):
    '''generator to return all possible variations where the section between two cities are swapped'''
    for i,j in all_pairs(len(tour)):
        if i != j:
            copy=tour[:]
            if i < j:
                copy[i:j+1]=tour[i:j+1][::-1]
            else:
                copy[i+1:] = tour[:j][::-1]
                copy[:j] = tour[i+1:][::-1]
            if not np.array_equal(copy, tour): # no point returning the same tour
                yield copy

def swapped_cities(tour):
    '''generator to create all possible variations where two cities have been swapped'''
    for i,j in all_pairs(len(tour)):
        if i < j:
            copy=tour[:]
            copy[i],copy[j]=tour[j],tour[i]
            yield copy

def cartesian_matrix(coords):
    '''create a distance matrix for the city coords that uses straight line distance'''
    matrix={}
    for i,(x1,y1) in enumerate(coords):
        for j,(x2,y2) in enumerate(coords):
            dx,dy=x1-x2,y1-y2
            dist=sqrt(dx*dx + dy*dy)
            matrix[i,j]=dist
    return matrix

def read_coords(coord_file):
    '''
    read the coordinates from file and return the distance matrix.
    coords should be stored as comma separated floats, one x,y pair per line.
    '''
    coords=[]
    for line in coord_file:
        x,y=line.strip().split(',')
        coords.append((float(x),float(y)))
    return coords

def tour_length(matrix,tour):
    tour = np.array(tour, int)
    
    '''total up the total length of the tour based on the distance matrix'''
    total=0
    num_cities=len(tour)
    for i in range(num_cities):
        j=(i+1)%num_cities
        city_i=tour[i]
        city_j=tour[j]
        total+=matrix[city_i,city_j]
    return total




def write_tour_to_img(coords,tour,title,img_file):
    padding=20
    # shift all coords in a bit
    coords=[(x+padding,y+padding) for (x,y) in coords]
    maxx,maxy=0,0
    for x,y in coords:
        maxx=max(x,maxx)
        maxy=max(y,maxy)
    maxx+=padding
    maxy+=padding
    img=Image.new("RGB",(int(maxx),int(maxy)),color=(255,255,255))
    
    font=ImageFont.load_default()
    d=ImageDraw.Draw(img);
    num_cities=len(tour)
    for i in range(num_cities):
        j=(i+1)%num_cities
        city_i=tour[i]
        city_j=tour[j]
        x1,y1=coords[city_i]
        x2,y2=coords[city_j]
        d.line((int(x1),int(y1),int(x2),int(y2)),fill=(0,0,0))
        d.text((int(x1)+7,int(y1)-5),str(i),font=font,fill=(32,32,32))
    
    
    for x,y in coords:
        x,y=int(x),int(y)
        d.ellipse((x-5,y-5,x+5,y+5),outline=(0,0,0),fill=(196,196,196))
    
    d.text((1,1),title,font=font,fill=(0,0,0))
    
    del d
    img.save(img_file, "PNG")

def init_random_tour(tour_length):
   tour=np.arange(tour_length).tolist()
   random.shuffle(tour)
   return tour

def run_hillclimb(init_function,move_operator,objective_function,max_iterations):
    from hillclimb import hillclimb_and_restart
    iterations,score,best=hillclimb_and_restart(init_function,move_operator,objective_function,max_iterations)
    return iterations,score,best

def run_anneal(init_function,move_operator,objective_function,max_iterations,start_temp,alpha, p=None):
    if start_temp is None or alpha is None:
        usage();
        print("missing --cooling start_temp:alpha for annealing")
        sys.exit(1)
    from sa import anneal
    iterations,score,best=anneal(init_function,move_operator,objective_function,max_iterations,start_temp,alpha, p)
    return iterations,score,best

def usage():
    print("usage: python %s [-o <output image file>] [-v] [-m reversed_sections|swapped_cities] -n <max iterations> [-a hillclimb|anneal] [--cooling start_temp:alpha] <city file>" % sys.argv[0])

def main(arg, p = None):
    max_iterations=10000

    move_operator=reversed_sections
    #move_operator=swapped_cities
    
    start_temp=10
    alpha=0.99995
    #start_temp,alpha=None,None

    def run_anneal_with_temp(init_function,move_operator,objective_function,max_iterations, p=None):
        return run_anneal(init_function,move_operator,objective_function,max_iterations,start_temp,alpha, p)
    run_algorithm=run_anneal_with_temp#run_hillclimb

    
    if isinstance(arg, dict):
        matrix = arg
        tmp = (1 + sqrt(1+4*len(arg))) / 2
        lc = np.round(tmp)
        assert -0.00001 < tmp - lc < 0.00001, 'matrix seems to have incorrect size'
#        assert matrix.ndim == 2, 'matrix of distances must have dimension 2'
#        assert matrix.shape[0] == matrix.shape[1], 'square matrix is expected'
#        lc = matrix.shape[0]
    else:
        city_file = arg
        F = open(city_file)
        coords=read_coords(F)
        matrix=cartesian_matrix(coords)
        lc = len(coords)
        
    init_function=lambda: init_random_tour(lc)
#    if iterfcn is None:
    objective_function=lambda tour: -tour_length(matrix,tour)
#    else:
#        def objective_function(tour):
#            r = -tour_length(matrix,tour)
#            objective_function.iterfcn_counter += 1
#            if not objective_function.iterfcn_counter % 1024:
#                iterfcn(np.array(tour), r)
#            return r
#        objective_function.iterfcn_counter = 0
    if p is not None:
        iterations,score,best=run_algorithm(init_function,move_operator, objective_function, p.maxFunEvals+10, p)
    else:
        iterations,score,best=run_algorithm(init_function,move_operator,objective_function, max_iterations)
    # output results
    return (iterations,score,best)    

if __name__ == "__main__":
    from time import time
    t = time()
    city_file = '/home/dmitrey/tsp/part_three/city100.txt'
    r = main(city_file)
    print(r)
    print('time elapsed: %0.1f' %(time()-t))

