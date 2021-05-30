import numpy as np
import skimage.measure
from scipy.special import softmax

def fitness(X_adv,X_true, label, model):
    """
    The fitness function
    
    X_adv -  the potential adversarial sample
    X_true - the original sample
    label - the true label of the original sample
    model - the model being attacked
    
    returns fitness of X_adv
    
    NOTE- This function will need to be changed depending on
          how we call the model to evaluate the samples
    
    We can also add a regularization term here, though it's unclear whether or not the paper uses it
    """
    ### Here evaluate our sample and get back the distances/simlarities, as well as the prediction###
    ### Change this depending on how we call our model to run predictions###
    dists, pred = evaluate(model,X_adv)
    f_true = dists[label]
    
    ###Assumes that we get back the distances/similarities as a list or np.array###
    best = np.max(np.append(dists[:label], dists[label+1:]) - f_true)
    
    ###TODO? Add Regularization term###
    
    ###If the model returns distances , this  is return -1*best. If it's similarities, this is return best###
    return -1*best

def CGC(parent_1, parent_2, cross_p, mut_p, s_max, beta=0):
    """
    The CGC algorithm
    
    parent_1 - the parent with a higher fitness score
    parent_2 - the parent with a lower fitness score
    
    cross_p -  the crossover probability for parent_1. 
                The cross over probabiity for parent_2 is 1 - cross_p
                
    mut_p -  the probability of mutation occuring
    
    s_max -  the maximum noise
    
    beta -  minimum threshhold for a gene to be considered "critical"
    
    returns the generated child
    """
    # Copy and reshape the parent into the original image shape
    child  = parent_1.reshape(28,28)
    
    # Max pooling, default 2x2
    pooled = skimage.measure.block_reduce(child,(2,2),np.max)
    
    #Upscale to original image size and flatten
    pooled = pooled.repeat(2,axis=0).repeat(2,axis=1).flatten()
    
    #normalize
    pooled = (pooled - np.min(pooled))/np.max(pooled)
    
    #Determine critical genes
    critical_genes = pooled > beta
    
    #Perform Crossover on critical genes
    mask = np.random.choice([0,1],size=X.shape,p=[1 - cross_p,cross_p])
    crossover = critical_genes*(mask*parent_1 + (1-mask)*parent_2)
    child = (pooled == 0)*child.flatten() + crossover
    
    #Add noise to critical genes
    mut = noise(parent_1, mut_p, s_max)
    child += critical_genes*mut
    return child

def noise(X, mut_p,s_max):
    """
    Generates uniformly distributed noise for a sample X
    
    X - the sample 
    mut_p - the probability of adding noise to an index
    s_max -  the maximum noise
    
    returns the noise. Must be added to the original sample.
    
    """
    noise = np.random.choice([i for i in range(-1*s_max,s_max)],size=X.shape)
    mask = np.random.choice([0,1],size=X.shape,p=[1 - mut_p,mut_p])
    return mask*noise

def pertAdj(X_adv,X_true,y,model):
    """
    Perturbation adjustment.
    
    X_adv - the adversarial sample
    X_true - the original sample
    y - the true label of the original sample
    model - the model being attacked
    
    returns the Adjusted adversarial sample
    """
    #Find Ids that don't match the original
    idxs = np.where(X_adv != X_true)[0]
    
    for idx in idxs:
        orig = X_true[idx]
        adv = X_adv[idx]
        
        # Determine which direction to iterate in
        if orig > adv:
            i = -1
        else:
            i = 1
        X_adv[idx] = orig
        
        ###Change this based on how we call the model to do predictions###
        dist,label = evaluate(model, X_adv)
        
        #Iterate until the label changes
        while label == y:
            X_adv[idx] += i

            ### Change to the same as above###
            dist,label = evaluate(model, X_adv)
    return X_adv

def adv_attack(model, X, y, N, i_max=1000,alg='GA-CGC',PA=True):
    """

    The main function for adversarial attacks

    Call this function to find an adversarial sample for a model

    ------------Input-------------
    model: The model being attacked
    X: The initial sample/datapoint
    y: The true label of X
    N: The population size of the algorithm
    i_max: Maximum amount of iterations to run the algorithm for
    alg: What version of the algorithm to use.
            GA - The basic Genetic algorithm
            GA-CGC - The Genetic algorithm with Critical Gene Crossover
            
    PA: Determines whether or not to use Perturbation adjustment
    
    ------------Output-------------
    
    (Boolean,Vector)
    
    Boolean - True if we have successfully found an adversarial sample, False if we have not
    
    Vector - The Adversarial sample if we have found one, else the closest sample we have
    
    """
    
    
    ### Sigma max, the maximum perturbation value ###
    s_max = int(np.max(X)*0.15)
    
    ### The chance of a gene mutating ###
    mut_p=0.05
    
    population = []
    next_pop = []
    indexes = [i for i in range(N)]
    
    # Initialize the population
    for i in range(N):
        if alg == 'GA-CGC':
            child = CGC(X, X, 1, 1,s_max)
        else:
            child = np.copy(X) + noise(X,1,s_max)
        population.append(child)
        
    #Run the genetic algorithm for i_max iterations
    successful = False
    for i in range(i_max):
        
        #find fitness score for each sample in the population
        scores = [fitness(sample,X,y,model) for sample in population]
        
        #Save the best sample for the next generation
        eli = population[np.argmax(scores)]
        next_pop.append(eli)
        dists,pred = evaluate(model,eli)
        if pred != y:
            successful = True
            break
        sel_p = softmax(scores)
        
        #Generate next generation
        for num in range (1, N):
            parents = np.random.choice(indexes,size=2,p=sel_p)
            if alg == 'GA':
                
                #Basic GA Algortithm
                p = scores[parents[0]]/(scores[parents[0]] + scores[parents[1]])
                mask = np.random.choice([0,1],size=X.shape,p=[1 - p,p])
                child = mask*population[parents[0]] + (1 - mask)*population[parents[1]]
                child += noise(child,mut_p,s_max)
            if alg == 'GA-CGC':
                
                #Find the most probable parent, then use CGC function to find children
                if scores[parents[0]] > scores[parents[1]]:
                    p = scores[parents[0]]/(scores[parents[0]] + scores[parents[1]])
                    child = CGC(population[parents[0]],population[parents[1]],p,mut_p,s_max)
                else:
                    p = scores[parents[1]]/(scores[parents[0]] + scores[parents[1]])
                    child = CGC(population[parents[1]],population[parents[0]],p,mut_p,s_max)
                    
            #Clip child to fit within proper values
            child = np.clip(child,np.min(X),np.max(X))
            next_pop.append(child)
        population = next_pop
        next_pop=[]
    if successful:
        eli = np.clip(eli,0,255)
        if PA:
            eli = pertAdj(eli,X,y,model)
        return successful, eli
    else:
        return successful, eli
                