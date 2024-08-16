# Author: Allison H. Moore, 
# Comments, questions and corrections to ahm6@rice.edu

# The purpose of this script is to compute the Alexander polynomial via the Kauffman state sum set of generators, realized as spanning trees of the black and white graphs associated with a checkerboard coloring of the knot. The algorithm is based on the classic Matrix Tree Theorem from graph theory and linear algebra.

from itertools import combinations
from copy import copy
from numpy import *

###################### INPUT ################################################
black_directed = array( [ (1,0,1,-1), (1,1,0,0), (0,1,-1,1) ] )
white_directed = array( [ (-1,1,1,0), (0,0,1,1), (1,-1,0,1) ] )
black_edgeweights = [0, 0, -1, -1]
white_edgeweights = [1, 1, 0, 0]
black_root = 1
white_root = 2
############################################################################

def modifyIncidence(incidence):
    """
    Takes in an array whose entires are assumed to be zeros and ones (e.g. an incidence matrix and returns where the first instance of +1 in each column has been changed to -1.
    """
    matrix = copy(incidence)
    n, m = matrix.shape
    for i in range(m):
        for j in range(n):
            if matrix[j,i] == 1:
                matrix[j,i] = -1
                break
    return matrix

def checkIncidence(incidence):
    """
    Checks to see whether the modified incidence matrix computed in the function modifyIncidence actually correpsonds to a bonafide incidence matrix or not
    """
    p, q = incidence.shape
    return all( sum(incidence, axis = 0) == zeros(q) )

def checkWeights(a, b, m):
    """
    Given two lists, it verifies that they have length m
    """
    return ( len(a) == m  and  len(b) == m )

def checkTree(modified, tup):
    """
    Given a incidence with n rows a tuple of length n-1, this checks whether the n x n-1 submatrix correspoding the the n-1 columns of the tuple represents a spanning tree or a cyclic spanning graph. We delete the first row of the modified incidence matrix and the determinant of the corresponding square submatrix. This corresponds with both a subgraph and one of the terms of the Cauchy Biney expansion for the determinant --- the determinant is one if and only if the subgraph corresponds exactly to a spanning tree.
    """
    n, m = modified.shape
    return ( abs(linalg.det(modified[1:n, tup])) == 1 )

def constructSpanningTree(incidence, tup):
    """
    Given an incidence matrix with n rows and a tuple of length n-1 which corresponds with a spanning tree, this returns an incidence matrix which describes exactly that spanning tree as a subgraph of the original graph.
    """
    n, m = incidence.shape
    # Replace the first row, and then replacing all entires in the deselected columns with zeros
    deselected = [ x for x in range(m) if x not in tup ]
    tree = copy( incidence )
    for j in deselected:
        tree[ 0:n, j ] = 0
    return tree, deselected

def constructDualTree(incidence, white_edges, m):
    """
    Given a spanning tree corresponding to the black graph associated with a checkerboard coloring of a knot, and a list corresponding to the indices of the edges not included in the black spanning tree, this constructs the dual spanning tree of the white graph.
    """
    n, m = incidence.shape
    deselected = list( set( range(m) ) - set( white_edges ) )
    tree = copy( incidence[ 0:n, ] )
    for j in deselected:
        tree[0:n, j] = 0
    return tree

def IncidenceToAdjacency(incidence):
    "Given the incidence matrix of an unweighted graph, this returns the adjacency matrix"
    laplacian = dot( abs( incidence ), incidence.transpose() )
    v, w = laplacian.shape
    for i in range(v):
        laplacian[i,i]=0
    return laplacian

def weightedMatrix(matrix, weights):
    """
    Given an nxm matrix and a length m vector indicating weights, this returns a copy of the matrix which is weighted by the column weights.
    """
    p, q = matrix.shape
    weighted = copy(matrix)
    for i in range(q):
        weighted[:, i] = weights[i]*weighted[:, i]
    return weighted

def rootFlow(adj_tree, root):
    """
    Given an adjacency matrix corresponding to a tree and an index corresponding to the root vertex, this function returns an antisymmetric adjacency matrix for this tree, where the signs of the entries record the flow induced on the tree by the root. For example if e = ij, and the flow points from i to j, then the if-entry of the matrix is +1 and the ji-entry is -1.
    
    Warning: if adj_tree is not a tree, this while loop may not terminate.
    """
    n, m = adj_tree.shape
    marked = [root]
    vertices = list( set(range(n)) - set(marked) )
    while len(vertices) > 0:
        new_marks = []
        for i in marked:
            for j in vertices:
                if adj_tree[i, j] !=0:
                    adj_tree[i, j] = -1* adj_tree[i, j]
                    new_marks.append(j)
            marked = new_marks        
            vertices = list( set(vertices) - set(marked) )
    return -1 * adj_tree

def computeGradings(tree, weights, directed, root):
    """
    Given an adjacency matrix corresponding to a spanning tree and the weight vector of the of the graph that the tree spans, this returns an integer corresponding to the A-grading of the Kauffman state determined by this tree.
    """
    n, m = tree.shape
    # Form a weighted adjacency matrix for the tree. Note this doesn't make sense for cyclic graphs. 
    weighted = weightedMatrix( tree, weights ) 
    #print 'weighted\n', weighted
    eta = IncidenceToAdjacency( weighted ) 
    #print 'eta\n', eta 
    
    # Form the flow orientation on the tree
    adj = IncidenceToAdjacency( tree )
    #print 'adj\n', adj
    flow_orientation = rootFlow( adj, root )
    #print 'flow_orientation\n', flow_orientation
    
    # Form the knot orientation
    directed = tree * directed
    knot_orientation = IncidenceToAdjacency( directed )
    #print 'knot_orientation\n', knot_orientation
    
    # See where the two orientations agree or disagree
    agree = ( knot_orientation == flow_orientation )
    #print 'agree\n', agree
    r, s = agree.shape
    
    # In order to change the mismatch weight from FALSE to -1 and change match weight from TRUE to +1 we use (2*agree - 1) when we take the weighted sum of sigma*eta over all the edges in the tree.
    # Alexander grading
    a_grading = 0
    for i in range(0, r):
        for j in range(i+1, r):
            a_grading = a_grading + ( 2*agree[i, j] -1 ) * eta[i, j]
            #print 'agree[i,j]', agree[i,j], 'plus eta[i,j]', eta[i,j]
            #print 'i,j', i,j, ' and grading', grading
    # Maslov grading
    m_grading = 0
    for i in range(0, r):
        for j in range(i+1, r):
            m_grading = m_grading + agree[i, j] * eta[i, j]
            #print 'agree[i,j]', agree[i,j], 'plus eta[i,j]', eta[i,j]
            #print 'i,j', i,j, ' and grading', m_grading
    return a_grading, m_grading

def get_spanning_trees(generators):
    """
    Given the dictionary generators, prints the incidence matrices (which are python arrays) of the trees which correspond with the keys of the dictionary
    """
    for tup in generators.keys():
        black_tree, white_edges = constructSpanningTree(incidence_b, i)
        white_tree = constructDualTree(incidence_w, white_edges, m)
        print 'black tree is \n', black_tree
        print 'white tree is \n', white_tree
    print

def get_max_A(generators):
    """
    Given the generator dictionary, this returns the integer value which is the maximum A-grading of the generators and the number of dictionary entries with that A-grading. Note that the A-grading is first entry in the grading tuple that is the key.    
    """
    max_key = max(generators.iterkeys(), key=(lambda key: generators[key]))
    max_value = generators[max_key][0]
    number = sum(1 for x in generators.values() if x[0] == max_value)
    return max_value, number

def get_min_A(generators):
        """
        Given the generator dictionary, this returns the integer value min_value which is the minimum A-grading of the generators and the number of dictionary entries with that A-grading as number. Note that the A-grading is first entry in the grading tuple that is the key.    
        """
        min_key = min(generators.iterkeys(), key=(lambda key: generators[key]))
        min_value = generators[min_key][0]
        number = sum(1 for x in generators.values() if x[0] == min_value)
        return min_value, number

def make_alexander_polynomial(generators):
    """
    Given the generator dictionary, this returns a string representing the Alexander polynomial of the knot along with a boolean indicating whether the polynomial is monic.
    """
    min_A = get_min_A(generators)
    # max_A = get_max_A(generators)
    # This constructs a new dictionary whose entries are the coefficients of the Alexander polynomial. The key is the A-grading and the value is the mod 2 sum of M-gradings of generators with that A-grading (i.e. the Euler characteristic).
    coefficients = {}
    for key in generators:
        A_grading = generators[key][0]
        M_grading = generators[key][1]
        # If the key  is not in the dict, it uses zero
        coefficients[A_grading] = int( (coefficients.get(A_grading) or 0 ) + (-1)**(M_grading) )
    first_coeff = abs(sorted(coefficients)[0])
    # Now, create a string to make it appear as an Alexander polynomial
    alex_poly = ""
    for i in sorted(coefficients):
        if i == 0:
            alex_poly = alex_poly + str(coefficients[i]) + ' + '
        else:
            alex_poly = alex_poly + str(coefficients[i]) + 't^('+ str(i) + ') + '
    if alex_poly.endswith(' + '):
        alex_poly = alex_poly[:-3] 
    return alex_poly, (first_coeff == 1)

def main():
    
    # Modify the incidence matrix to be able to apply the proof of the Matrix Tree Theorem
    incidence_b, incidence_w = abs(black_directed), abs(white_directed)
    n, m = incidence_b.shape
    modified_b, modified_w = modifyIncidence(incidence_b), modifyIncidence(incidence_w)

    # Check that the original matrix corresponds with a bonafide incidence matrix
    if ( checkIncidence(modified_b) and checkIncidence(modified_w) ):
        print 'Valid incidence matrices.'
    elif (not checkIncidence(modfied_b) ):
        print 'error: black_directed is not a valid incidence matrix'
    else:
        print 'error: white_directed is not a valid incidence matrix'
    # Check that the weight lists are the correct length.
    if checkWeights( black_edgeweights, white_edgeweights, m ):
        print 'Valid length weight lists.'
    else:
        print 'error: weight lists not the correct length'

    # Compute the set of black spanning trees and their dual spanning trees, and for each pair compute the A and M-gradings of the corresponding Kauffman state.
    generators = {}
    for i in combinations(range(m), n-1):
        if checkTree(modified_b, i):
            black_tree, white_edges = constructSpanningTree(incidence_b, i)
            white_tree = constructDualTree(incidence_w, white_edges, m)
        else:
            continue
        a_grading_b, m_grading_b = computeGradings( black_tree, black_edgeweights, black_directed, black_root )
        a_grading_w, m_grading_w = computeGradings( white_tree, white_edgeweights, white_directed, white_root )
        generators[i] = ( (a_grading_b + a_grading_w) / 2, m_grading_b + m_grading_w )

    ##### Test statements #####
        # See the black and white spanning trees:
        # print 'black spanning tree\n', black_tree
        # print 'white spanning tree\n', white_tree
        # List the spanning trees
        # get_spanning_trees(generators)
    ############################

    # Determine the maximum A-grading and the number of states in that max A-grading. Do the same for the minimum A-grading. Then compute the Alexander polynomial and determine if monic.
    max_A_grading, number_in_max = get_max_A(generators)
    min_A_grading, number_in_min = get_min_A(generators)
    alexander_polynomial, isMonic = make_alexander_polynomial(generators)

    print 'The maximum A-grading is ', max_A_grading, ' and number in max A-grading is', number_in_max
    print 'The minumum A-grading is ', min_A_grading, ' and number in min A-grading is', number_in_min
    print 'Alexander polynomial is ', alexander_polynomial

    if ( number_in_max == 1 or number_in_min == 1):
        print 'Your knot is fibered because the unique state property is satisfied.'
    elif isMonic:
        print 'The unique state property is not satisfied, yet the Alexander polynomial is monic, hence you knot may or may not be fibered. Try changing the root selection or draw another diagram.'
    elif not isMonic:
        print 'Your knot is not fibered because the Alexander polynomial is not monic.'
    else:
        print 'Something went wrong.'
        
if __name__ == "__main__":
    main()
