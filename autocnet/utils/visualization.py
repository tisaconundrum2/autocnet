import autocnet
import math # plotAdjacencyGraphFeatures
import matplotlib # plotting
import numpy as np # plotFeatures

from autocnet.examples import get_path # get file path
from autocnet.fileio.io_gdal import GeoDataset # set handle, get image as array
from matplotlib import pyplot as plt # plotting 


def plotFeatures(imageName, keypoints, pointColorAndHatch='b.', featurePointSize=7):
    """
    Plot an image and its found features using the image file name and 
    a keypoint list found using autocnet.feature_extractor. The user may
    also specify the color and style of the points to be plotted and the
    size of the points.

    Parameters
    ----------
    imageName : str
                The base name of the image file (without path). 
                This will be the title of the plot.

    keypoints : list
                The keypoints of this image found by the feature extractor.

    pointColorAndHatch : str
                         The color and hatch (symbol) to be used to mark
                         the found features. Defaults to 'b.', blue and 
                         square dot. See matplotlib documentation for
                         more choices.

    featurePointSize : int
                       The size of the point marker. Defaults to 7.

    """

    imgArray = GeoDataset(get_path(imageName)).read_array()
    height, width = imgArray.shape[:2]

    displayBox = np.zeros((height, width), np.uint8)
    displayBox[:height, :width] = imgArray

    plt.title(imageName)
    plt.margins(tight=True)
    plt.axis('off')
    plt.imshow(displayBox, cmap='Greys')

    for kp in keypoints: 
        x,y = kp.pt
        plt.plot(x,y,pointColorAndHatch, markersize=featurePointSize)
        


def plotAdjacencyGraphFeatures(graph, pointColorAndHatch='b.', featurePointSize=7):
    """
    Plot each image in an adjacency graph and its found features in a single figure.
    The user may also specify the color and style of the points to be plotted and
    the size of the points.

    Parameters
    ----------
    graph : object
            A CandicateGraph object from a JSON file whose node (i.e. image)
            keypoints have been filled using the autocnet.feature_extractor.

    pointColorAndHatch : str
                         The color and hatch (symbol) to be used to mark
                         the found features. Defaults to 'b.', blue and 
                         square dot. See matplotlib documentation for
                         more choices.

    featurePointSize : int
                       The size of the point marker. Defaults to 7.

    """

    counter = 1
    columns = math.ceil(math.sqrt(graph.number_of_nodes()))
    rows = columns
    for node, attributes in graph.nodes_iter(data=True):
        plt.subplot(rows, columns, counter)
        plotFeatures(attributes['handle'], 
                         attributes['keypoints'], 
                         pointColorAndHatch)
        counter = counter + 1

def plotAdjacencyGraphMatches(imageName1, 
                              imageName2, 
                              graph,
                              aspectRatio=0.44,
                              featurePointSize=3,
                              lineWidth=1,
                              lineColor='g',
                              saveToFile='FeatureMatches.png'):
    """
    Plot the features of two images in the given adjacency graph and draw lines
    between matching features. The user may specify the size of the points and 
    lines to be plotted, but the colors and styles of the points are internally
    set to green circles ('go') for matched features and red circles ('ro') for
    unmatched features. The output plot is saved under the given file name
    in order to lock the aspect of the plotted lines.

    Parameters
    ----------
    imageName : str
                The name of the first image file (with extension, without path).
                This will be the title of the left subplot.

    imageName : str
                The name of the second image file (with extension, without path).
                This will be the title of the right subplot.

    graph : object
            A CandicateGraph object containing the given images (as nodes) and their
            matches (edges). This graph is read from a JSON file, autocnet.feature_extractor
            has been applied, and FlannMatcher has been applied.

    aspectRatio : float
                  The ratio of the height/width of the figure.

    featurePointSize : int
                       The size of the feature point marker. Defaults to 3.

    lineWidth : int
                The width of the match lines. Defaults to 1.

    lineColor : str
                Color code for the matching lines. Defaults to 'g' (green).

    saveToFile : str
                 A file name to which the figure is saved. Defaults to 'FeatureMatches.png'.
                 If saveToFile='', then the figure will be shown, but lines will not be
                 locked.
    """

    w,h = matplotlib.figure.figaspect(aspectRatio)    
    fig = plt.figure(figsize=(w,h))     

    columns = 2
    rows = 1
    # create a subplot with its own axes
    ax1 = fig.add_subplot(rows, columns, 1)
    ax1.axis('off')
    ax1.margins(tight=True)
    plotFeatures(imageName1, 
                 graph.get_keypoints(imageName1), 
                 'ro', 
                 featurePointSize)
    ax2 = fig.add_subplot(rows, columns, 2)
    ax1.axis('off')
    ax2.margins(tight=True)
    plotFeatures(imageName2, 
                 graph.get_keypoints(imageName2), 
                 'ro', 
                 featurePointSize)
    edge = graph[graph.node_name_map[imageName1]][graph.node_name_map[imageName2]]
    if 'matches' in edge.keys():
        for i, row in edge['matches'].iterrows():
            # get matching points
            image1ID = int(row['source_idx'])
            image2ID = int(row['destination_idx'])
            keypointImage1 = (graph.get_keypoints(imageName1)[image1ID].pt[0],
                              graph.get_keypoints(imageName1)[image1ID].pt[1])
            keypointImage2 = (graph.get_keypoints(imageName2)[image2ID].pt[0],
                              graph.get_keypoints(imageName2)[image2ID].pt[1])
            
            # change points to green
            ax1.plot(keypointImage1[0], keypointImage1[1], 'go', markersize=featurePointSize)
            ax2.plot(keypointImage2[0], keypointImage2[1], 'go', markersize=featurePointSize)

            # Use the transform() method to get the display coordinates
            # from the data coordinates of the matching points in each subplot
            displayCoord1 = ax1.transData.transform([keypointImage1[0],
                                                     keypointImage1[1]])
            displayCoord2 = ax2.transData.transform([keypointImage2[0],
                                                     keypointImage2[1]])
            # Use the inverted() method to create a transform that will convert
            # our display coordinates to data coordinates for the entire figure.
            inv = fig.transFigure.inverted()
            coord1 = inv.transform(displayCoord1)
            coord2 = inv.transform(displayCoord2)

            # construct a line between the matching points using the data coordinates and the 
            # transformation from data coordinates to display coordinates
            line = matplotlib.lines.Line2D((coord1[0], coord2[0]), 
                                           (coord1[1], coord2[1]),
                                           transform=fig.transFigure,
                                           color=lineColor, 
                                           marker='o',
                                           markeredgecolor='g',
                                           markersize=featurePointSize,
                                           linewidth=lineWidth,
                                           alpha=0.5)
            fig.lines.append(line)

    plt.axis('off')
    plt.margins(tight=True)

    if saveToFile:
        fig.savefig(saveToFile)
        plt.close()
    else:
        plt.show()


# NOTE: We will probably delete this code if it is found to be un-needed,
# However, for now we will keep in case it winds up being a more useful tool.
def plotAdjacencyGraphMatchesSingleDisplay(imageName1, 
                                           imageName2, 
                                           graph,
                                           featurePointSize=10,
                                           lineWidth=3):
    """
    This is an earlier version of plotAdjacencyGraphMatches() where the
    images are offset in a single display box rather than in their
    own subplots.

    Parameters
    ----------
    imageName : str
                The name of the first image file (with extension, without path).
                This will be the title of the left subplot.

    imageName : str
                The name of the second image file (with extension, without path).
                This will be the title of the right subplot.

    graph : object
            A CandicateGraph object containing the given images (as nodes) and their
            matches (edges). This graph is read from a JSON file, autocnet.feature_extractor
            has been applied, and FlannMatcher has been applied.

    featurePointSize : int
                       The size of the feature point marker. Defaults to 10.

    lineWidth : int
                The width of the match lines. Defaults to 3.

    Returns
    -------
     : AxesImage object
       An image object that can be saved. 
    """

    imgArray1 = GeoDataset(get_path(imageName1)).read_array()
    imgArray2 = GeoDataset(get_path(imageName2)).read_array()
    
    height1, width1 = imgArray1.shape[:2]
    height2, width2 = imgArray2.shape[:2]
    
    w = width1+width2+50
    h = max(height1, height2)
    
    displayBox = np.zeros((h, w), np.uint8)
       
    displayBox[:height1, :width1] = imgArray1
    displayBox[:height2, width1+50:w] = imgArray2
    
    for kp in graph.get_keypoints(imageName1): 
        x, y = kp.pt
        plt.plot(x, y,'ro', markersize=featurePointSize)
    for kp in graph.get_keypoints(imageName2): 
        x, y = kp.pt
        plt.plot(x+width1+50, y,'ro', markersize=featurePointSize)
    
    edge = graph[graph.node_name_map[imageName1]][graph.node_name_map[imageName2]]
    if 'matches' in edge.keys():
        for i, row in edge['matches'].iterrows():
            # get matching points
            image1ID = int(row['source_idx'])
            image2ID = int(row['destination_idx'])
            keypointImage1 = (graph.get_keypoints(imageName1)[image1ID].pt[0],
                              graph.get_keypoints(imageName1)[image1ID].pt[1])
            keypointImage2 = (graph.get_keypoints(imageName2)[image2ID].pt[0]+width1+50,
                              graph.get_keypoints(imageName2)[image2ID].pt[1])
                            
            # construct a line between the matching points using the data coordinates and the 
            # transformation from data coordinates to display coordinates
            plt.plot([keypointImage1[0], keypointImage2[0]], 
                     [keypointImage1[1], keypointImage2[1]],
                     color='g', 
                     marker='o',
                     markeredgecolor='g',
                     markersize=featurePointSize,
                     linewidth=lineWidth,
                     alpha=0.5)
    return plt.imshow(displayBox, cmap='Greys')
