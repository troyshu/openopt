#from numpy import asfarray, argmax, sign, inf, log10
from openopt.kernel.baseSolver import baseSolver
from numpy import asfarray,  inf,  atleast_1d
from openopt.kernel.setDefaultIterFuncs import SMALL_DELTA_X,  SMALL_DELTA_F

class galileo(baseSolver):
    __name__ = 'galileo'
    __license__ = "GPL"
    __authors__ = 'Donald Goodman, dgoodman-at-cs.msstate.edu, connected to OO by Dmitrey'
    __alg__ = "Genetic Algorithm, same as ga2, the C++ canonical genetic algorithm lib, also by Donald Goodman"
    iterfcnConnected = True
    __homepage__ = 'http://www.cs.msstate.edu/~dgoodman'

    __info__ = """
    requires finite lb, ub: lb <= x <= ub
    """
    __optionalDataThatCanBeHandled__ = ['lb', 'ub']
    __isIterPointAlwaysFeasible__ = lambda self, p: True

    population = 15
    crossoverRate = 1.0 # 1.0 means always
    mutationRate = 0.05 # not very often
    useInteger = False # use float by default
    _requiresFiniteBoxBounds = True


    def __init__(self):pass
    def __solver__(self, p):

        p.kernelIterFuncs.pop(SMALL_DELTA_X)
        p.kernelIterFuncs.pop(SMALL_DELTA_F)
        p.ff = inf

        #create an initial population of 10 chromosomes
        P = Population(self.population)# CHECKME! is the Population size optimal?

        #use fitness as our evaluation function
        P.evalFunc = lambda x: -p.f(x) # fitness

        #minimum values the genes can take
        P.chromoMinValues = p.lb.tolist()

        #maximum values the genes can take
        P.chromoMaxValues = p.ub.tolist()

        #use integers instead of floats
        P.useInteger = self.useInteger

        #crossover
        P.crossoverRate = self.crossoverRate

        #mutate, but not very often
        P.mutationRate = self.mutationRate

        #use roulette (monte carlo) selection
        P.selectFunc = P.select_Roulette

        #use a full replacement size
        P.replacementSize = P.numChromosomes

        #use one point crossover
        P.crossoverFunc = P.crossover_OnePoint

        #use the default mutation routines
        P.mutateFunc = P.mutate_Default

        #use steady-state replacement with no duplication
        P.replaceFunc = P.replace_SteadyStateNoDuplicates

        #finish initializing the population. THIS MUST BE CALLED after settings the
        #variables above, but before actually running the GA!
        P.prepPopulation()
        #for 50 epochs

        for itn in range(p.maxIter+1):
          #evaluate each chromosomes
          P.evaluate()
          #apply selection
          P.select()
          #apply crossover
          P.crossover()
          #apply mutation
          P.mutate()
          #apply replacement
          P.replace()
          #print the best fit individual, and its fitness
          fval = asfarray(-P.maxFitness)
          if p.ff > fval:
              p.xf,  p.ff = asfarray(P.bestFitIndividual.genes), fval
          p.iterfcn(asfarray(P.bestFitIndividual.genes), fval)

          if p.istop:
              #p.xk,  p.fk = p.xf,  p.ff
              return



from random import Random

class Chromosome:
  """The Chromosome class represents a single chromosome in a population.
  A Chromosome contains some number of genes (Python objects), and can
  be treated as a list, with indices and slices and all that good stuff
  """

  def __init__(self):
    """Constructs a new Chromosome instance"""
    self.genes = []
    self.geneMaxValues = []
    self.geneMinValues = []
    self.fitness = None
    self.evalFunc = None
    self.parent = None

  def __str__(self):
    return self.genes.__str__()

  def randomInit(self, generator, intValues = 1):
    """Randomly initializes all genes within the ranges set with
    setMinValues and setMaxValues. generator should be an instance
    of Random from the random module. Doing things this way allows for
    thread safety. If intValues is set to 1, random
    integers will be created, else random floating point values will
    be generated.
    """
    #first, are the lists empty?
    minlen = len(self.geneMinValues)
    maxlen = len(self.geneMaxValues)
    if (minlen == 0) or (minlen != maxlen):
      return
    randFunc = None
    if intValues == 1:
      randFunc = generator.randint
    else:
      randFunc = generator.uniform
    self.genes = []
    for i in range(minlen):
      self.genes.append(randFunc(self.geneMinValues[i], self.geneMaxValues[i]))

    self.fitness = None

  def evaluate(self):
    """Calls evalFunc for this chromosome, and caches the fitness value
    returned. Returns None if evalFunc is not yet defined.
    """
    if self.evalFunc != None:
      self.fitness = self.evalFunc(self.genes)
      return self.fitness
    else:
      return None

  def getFitness(self):
    """Calls evaluate if there is no cached value, otherwise returns the cached
    fitness value.
    """
    if self.fitness != None:
      return self.fitness
    else:
      return self.evaluate()

  def copy(self):
    """Duplicates the chromosome.
    """
    retval = Chromosome()
    for item in self.__dict__:
      retval.__dict__[item] = self.__dict__[item]
    return retval

  def __len__(self):
    return len(self.genes)

  def __getitem__(self, key):
    retval = self.copy()
    retval.genes = [self.genes[key]]
    retval.geneMinValues = [self.geneMinValues[key]]
    retval.geneMaxValues = [self.geneMaxValues[key]]
    retval.fitness = None
    return retval

  def __setitem__(self, key, value):
    return self.genes.__setitem__(key, value)

  def __getslice__(self, i, j):
    retval = self.copy()
    retval.genes = self.genes[i:j]
    retval.geneMinValues = self.geneMinValues[i:j]
    retval.geneMaxValues = self.geneMaxValues[i:j]
    retval.fitness = None
    return retval
    return self.genes.__getslice__(i, j)

  def __contains__(self, item):
    return self.genes.__contains__(item)

  def __add__(self, other):
    retval = self.copy()
    retval.genes = self.genes + other.genes
    retval.geneMinValues = self.geneMinValues + other.geneMinValues
    retval.geneMaxValues = self.geneMaxValues + other.geneMaxValues
    retval.fitness = None
    return retval

  def __cmp__(self, other):
    s1 = self.getFitness()
    s2 = other.getFitness()
    return s1-s2

  def isIdentical(self, other):
    """If the genes in self and other are identical, returns 0
    """
    return (self.genes == other.genes)

class Population:
  """The Population class represents an entire population of a single
  generation of Chromosomes. This population is replaced with each iteration
  of the algorithm. Functions are provided for storing generations for later
  analysis or retrieval, or for reloading the population from some point.
  All of the high level functionality is in this
  class: generally speaking, you will almost never call a function from any
  of the other classes.
  """

  def __init__(self, numChromosomes):
    """Constructs a population of chromosomes, with numChromosomes as the
    size of the population. Note that prepPopulation must also be called
    after all user defined variables have been set, to finish initialization.
    """
    self.numChromosomes = numChromosomes
    self.currentGeneration = []
    self.nextGeneration = []
    self.chromoMaxValues = []
    self.chromoMinValues = []

    self.mutationRate = 0.0
    self.crossoverRate = 0.0
    self.replacementSize = 0
    self.useInteger = 0
    self.isSorted = 0

    self.crossoverCount = 0
    self.mutationCount = 0

    self.evalFunc = None
    self.mutateFunc = None
    self.selectFunc = None
    self.crossoverFunc = None
    self.replaceFunc = None

    self.generator = Random()

    self.minFitness = None
    self.maxFitness = None
    self.avgFitness = None
    self.sumFitness = None
    self.bestFitIndividual = None

  def prepPopulation(self):
    """Radnomly initializes each chromosome according to the values in
    chromosMinValues and chromosMaxValues.
    """
    if (len(self.chromoMinValues) != len(self.chromoMaxValues)) or (len(self.chromoMinValues) == 0):
      return None

    self.currentGeneration = []

    for i in range(self.numChromosomes):
      c = Chromosome()
      c.geneMinValues = self.chromoMinValues
      c.geneMaxValues = self.chromoMaxValues
      c.randomInit(self.generator, self.useInteger)
      c.evalFunc = self.evalFunc
      self.currentGeneration.append(c)

    return 1

  def evaluate(self):
    """Evaluates each chromosome. Since fitness values are cached, don't
    hesistate to call many times. Also calculates sumFitness, avgFitness,
    maxFitness, minFitness, and finds bestFitIndividual, for your convienence.
    Be sure to assign an evalFunc
    """

    self.sumFitness = 0.0
    self.avgFitness = 0.0
    self.maxFitness = self.currentGeneration[0].getFitness()
    self.minFitness = self.currentGeneration[0].getFitness()
    self.bestFitIndividual = self.currentGeneration[0]

    for chromo in self.currentGeneration:
      f = chromo.getFitness()
      self.sumFitness = self.sumFitness + f
      if f > self.maxFitness:
        self.maxFitness = f
        self.bestFitIndividual = chromo
      elif f < self.minFitness: self.minFitness = f

    self.avgFitness = self.sumFitness/len(self.currentGeneration)

  def mutate(self):
    """At probability mutationRate, mutates each gene of each chromosome. That
    is, each gene has a mutationRate chance of being randomly re-initialized.
    Right now, only mutate_Default is available for assignment to mutateFunc.
    """

    self.mutationCount = 0
    for i in range(self.replacementSize):
      self.nextGeneration[i] = self.mutateFunc(self.nextGeneration[i])

  def select(self):
    """Selects chromosomes from currentGeneration for placement into
    nextGeneration based on selectFunc.
    """

    self.nextGeneration = []
    for i in range(0, self.replacementSize, 2):
      s1 = self.selectFunc()
      s2 = self.selectFunc()
      s1.parent = (s1, s1)
      s2.parent = (s2, s2)
      self.nextGeneration.append(s1)
      self.nextGeneration.append(s2)

  def crossover(self):
    """Performs crossover on pairs of chromos in nextGeneration with probability
    crossoverRate. Calls crossoverFunc, which must be set; current choices are
    crossover_OnePoint, crossover_TwoPoint and crossover_Uniform.
    """

    self.crossCount = 0
    for i in range(0, self.replacementSize, 2):
      (a,b) = self.crossoverFunc(self.nextGeneration[i], self.nextGeneration[i+1])
      (self.nextGeneration[i],self.nextGeneration[i+1]) = (a,b)

  def replace(self):
    """Replaces currentGeneration with nextGeneration according to the rules
    set forth in replaceFunc. Right now, replaceFunc can take the values of
    replace_SteadyState, replace_SteadyStateNoDuplicates and
    replace_Generational.
    """

    return self.replaceFunc()

  def select_Roulette(self):
    """Perform Roulette (Monte Carlo) selection. Assign this function to
    selectFunc to use.
    In essence, we construct a big roulette wheel, with a slot for each
    individual. The size of each slot is proportional to the relative fitness
    of that individual. The wheel is then spun! whee! The more fit individuals
    have a greater chance of landing under the pointer. The individual that
    lands under the pointer is returned.
    """

    partialSum = 0.0
    #spin the wheel!!
    wheelPosition = self.generator.uniform(0, self.sumFitness)
    i = 0
    for chromo in self.currentGeneration:
      partialSum = partialSum + chromo.getFitness()
      if partialSum >= wheelPosition:
        return chromo
      i = i + 1
    return self.currentGeneration[-1]

  def select_Ranked(self):
    """Currently does nothing. Hrm.
    """
    return None

  def crossover_OnePoint(self, chromo1, chromo2):
    """A crossover function that can be assigned to crossoverFunc. This one
    takes two chromosomes, cuts them at some random point, and swaps the parts
    creating two new chromosomes, which are returned in a tuple. Note
    that there is only a crossoverRate chance of crossover happening.
    """
    prob = self.generator.random()
    if prob <= self.crossoverRate:
      self.crossoverCount = self.crossoverCount + 1
      cutPoint = self.generator.randint(0, len(chromo1)-1)
      newchromo1 = chromo1[:cutPoint]+chromo2[cutPoint:]
      newchromo2 = chromo2[:cutPoint]+chromo1[cutPoint:]
      return (newchromo1, newchromo2)
    else:
      return (chromo1, chromo2)


    """A crossover function that can be assigned to crossoverFunc. This one
    takes two chromosomes, cuts them at two random points (creating three
    parts for each chromosomes), and swaps the parts around, creating two
    new chromosomes, which are returned in a tuple. Note
    that there is only a crossoverRate chance of crossover happening.
    """
    prob = self.generator.random()
    if prob <= self.crossoverRate:
      self.crossoverCount = self.crossoverCount + 1
      cutPoint1 = self.generator.randint(0, len(chromo1)-1)
      cutPoint2 = self.generator.randint(1, len(chromo1))
      if cutPoint2 < cutPoint1:
        temp = cutPoint1
        cutPoint1 = cutPoint2
        cutPoint2 = temp

      newchromo1 = chromo1[:cutPoint1]+chromo2[cutPoint1:cutPoint2]+chromo1[cutPoint2:]
      newchromo2 = chromo2[:cutPoint1]+chromo1[cutPoint1:cutPoint2]+chromo2[cutPoint2:]

      return (newchromo1, newchromo2)
    else:
      return (chromo1, chromo2)

  def crossover_Uniform(self, chromo1, chromo2):
    """A crossover function that can be assigned to crossoverFunc. Creates
    two new chromosomes by flippinng a coin for each gene. If the coin is heads,
    the gene values in chromo1 and chromo2 are swapped (otherwise they are
    left alone). The two new chromosomes are returned in a tuple. Note
    that there is only a crossoverRate chance of crossover happening.
    """

    prob = self.generator.random()
    if prob <= self.crossoverRate:
      self.crossoverCount = self.crossoverCount + 1
      newchromo1 = chromo1.copy()
      newchromo2 = chromo2.copy()
      for i in range(len(chromo1)):
        #flip a coin...1 we switch, 0 we do nothing
        coin = self.generator.randint(0,1)
        if coin == 1:
          temp = newchromo1.genes[i]
          newchromo1.genes[i] = newchromo2.genes[i]
          newchromo2.genes[i] = temp
      return (newchromo1, newchromo2)
    else:
      return (chromo1, chromo2)

  def mutate_Default(self, chromo):
    """Mutation function that can be assigned to mutateFunc. For each gene
    on each chromosome, there is a mutationRate chance that it will be
    randomly re-initialized. The chromosome is returned.
    """
    for i in range(len(chromo.genes)):
      prob = self.generator.random()
      if prob <= self.mutationRate:
        #then we mutate!
        self.mutationCount = self.mutationCount + 1
        f = 0
        if self.useInteger:
          f = self.generator.randint(self.chromoMinValues[i], self.chromoMaxValues[i])
        else:
          f = self.generator.uniform(self.chromoMinValues[i], self.chromoMaxValues[i])
        chromo.genes[i] = f
    return chromo

  def replace_SteadyState(self):
    """Replacement function that can be assigned to replaceFunc. Takes the
    values in nextGeneration, sticks them into currentGeneration, sorts
    currentGeneration, and lops off enough of the least fit individuals
    to reduce the size of currentGeneration back to numChromosomes.
    """

    for chromo in self.nextGeneration:
      self.currentGeneration.append(chromo)
    self.currentGeneration.sort()
    self.currentGeneration.reverse()
    self.currentGeneration = self.currentGeneration[:self.numChromosomes]
    self.nextGeneration = []

  def replace_SteadyStateNoDuplicates(self):
    """Replacement function that can be assigned to replaceFunc. Same as
    replace_SteadyState, exccept that duplicate chromosomes are not inserted
    back into the currentGeneration.
    """
    #this one is like above, but no duplicates are allowed!
    for chromo in self.nextGeneration:
      flag = 0
      for chromo2 in self.currentGeneration:
        if chromo.isIdentical(chromo2):
          flag = 1
      if flag == 0:
        self.currentGeneration.append(chromo)
    self.currentGeneration.sort()
    self.currentGeneration.reverse()
    self.currentGeneration = self.currentGeneration[:self.numChromosomes]
    self.nextGeneration = []

  def replace_Generational(self):
    """Replacement function that can be assigned to replaceFunc. Wholesale
    replacement of currentGeneration with nextGeneration. assumes that
    replacementSize is equal to numChromosomes; otherwise, the
    currentGeneration will shrink in size to replacementSize in size.
    """
    self.currentGeneration = self.nextGeneration[:]
    self.nextGeneration = []
