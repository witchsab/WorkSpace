import os
import pickle
from itertools import combinations

from imutils import paths


class AccuracyGlobal:
    '''
    This class contains all the accuracy measurement codes for any directory
    The loaded file contains a dict of { ID : [List of ID matches]}
    '''
    seed = []
    groundTruth = {}

    
    def read(self, dirpath='.'):
        ''' Initialize the files for search '''
        print('Reading directory', dirpath)
        imagepaths = sorted(list(paths.list_images(dirpath)))
        print('Found', len(imagepaths), 'files.')
        print("Loading Dict and Ground Truth.")
        # read pickle seed
        # read pickle groundTruth
        # reading the pickle tree
        infile = open(dirpath+'groundTruth.pickle','rb')
        self.groundTruth = pickle.load(infile)

    def check_ground_truth(self): 
        print ('Ground Truth Value', len(self.groundTruth))
        return self.groundTruth


    def accuracy_matches(self, q_path, imagematches, window):
        '''Gets Accuracy vs ground truth for search results in imagematches'''
        acc = 0
        queryfilename = os.path.basename(q_path)

        # print (queryfilename)
        # filenumber = int(queryfilename.split('.')[0].split('ukbench')[1])
        # fileext = queryfilename.split('.')[1]
        # start = filenumber - filenumber%4
        #init a list of match file names
        # familyfilename = []

        # for i in range(start, start+4):
        #     familyfilename.append('ukbench'+ str(i).zfill(5)+ '.' + fileext)
        # # print(familyfilename)
        
        # from dict
        familyfilename = self.groundTruth[queryfilename]
        # print ('query, family:', queryfilename, familyfilename)

        try:
            familyfilename.remove(queryfilename)
        except:
            pass

        #init search list for accuracy score calculation
        searchlist = []
        for item in imagematches[:window]:
            score, path = item
            searchlist.append(os.path.basename(path))
        # print (searchlist)
        try:
            searchlist.remove(queryfilename)
        except: 
            pass
        overlap = list(set(familyfilename).intersection(searchlist))
        # print ('overlap', overlap)
        acc = (len(overlap)/len(familyfilename)) *100
        # Accurracy metric calculation
        searchlist = []
        for item in imagematches:
            score, path = item
            searchlist.append(os.path.basename(path))
        # print (searchlist)
        try : 
            searchlist.remove(queryfilename)
        except: 
            pass
        overlap = list(set(familyfilename).intersection(searchlist))
        # print ('overlap', overlap)
        indexlist = []
        for common in overlap: 
            try: 
                indexlist.append(searchlist.index(common) + 1)
            except: 
                pass
        indexlist.append(0)
        sorted_position = sorted(indexlist)
        
        # print (sorted_position)
        metricValue = self.metricCalc(indexlist)
        # metricValue = metricCalc(list(range(0, len(familyfilename)+1))) /metricCalc(indexlist)
        # print (metricValue)
        # print (queryfilename)
        # print (searchlist)
        return (acc, metricValue, sorted_position, len(sorted_position) )

    def metricCalc(self, indexlist):
        '''Quality Calculator'''
        d = 0 
        for x,y in combinations(indexlist, 2): 
            d += abs(x-y) 
        # print ("Metric ", d)
        return d
    # accuracy_matches ('./imagesbooks/ukbench07994.jpg', imagematches, 50 )


    def accuracy_from_list(self, q_path, imagelist, window):
        acc = 0
        queryfilename = os.path.basename(q_path)
        # # print (queryfilename)
        # filenumber = int(queryfilename.split('.')[0].split('ukbench')[1])
        # fileext = queryfilename.split('.')[1]
        # start = filenumber - filenumber%4
        # #init a list of match file names
        # familyfilename = []

        familyfilename = self.groundTruth[queryfilename]

        try: 
            familyfilename.remove(queryfilename)
        except: 
            pass

        #init search list for accuracy score calculation
        searchlist = []
        for item in imagelist[:window]:
            
            searchlist.append(os.path.basename(item))
        # print (searchlist)
        try : 
            searchlist.remove(queryfilename)
        except: 
            pass
        overlap = list(set(familyfilename).intersection(searchlist))
        # print ('overlap', overlap)
        acc = (len(overlap)/len(familyfilename)) *100
        # Accurracy metric calculation 
        searchlist = []
        for item in imagelist:
            searchlist.append(os.path.basename(item))
        # print (searchlist)
        try : 
            searchlist.remove(queryfilename)
        except: 
            pass
        overlap = list(set(familyfilename).intersection(searchlist))
        # print ('overlap', overlap)
        indexlist = []
        for common in overlap: 
            try : 
                indexlist.append(searchlist.index(common) + 1)
            except : 
                pass
        indexlist.append(0)
        sorted_position = sorted(indexlist)
        
        # print (sorted_position)
        metricValue = self.metricCalc(indexlist)
        # metricValue = metricCalc(list(range(0, len(familyfilename)+1))) /metricCalc(indexlist)
        # print (metricValue)
        # print (queryfilename)
        # print (searchlist)
        return (acc, metricValue, sorted_position, len(sorted_position) )

    def accuracy_groundtruth(self, q_path):
        '''
        List of groundtruth array 
        '''
        queryfilename = os.path.basename(q_path)
        # # print (queryfilename)
        # filenumber = int(queryfilename.split('.')[0].split('ukbench')[1])
        # fileext = queryfilename.split('.')[1]
        # start = filenumber - filenumber%4
        # #init a list of match file names
        # familyfilename = []

        familyfilename = self.groundTruth[queryfilename]
        
        # for i in range(start, start+4):
        #     familyfilename.append(os.path.dirname(q_path) +'/ukbench'+ str(i).zfill(5)+ '.' + fileext)
        # # print(familyfilename)
    
        return familyfilename

    def accuracy_groundtruth_gen(self, q_path):
        '''
        Generates GroundTruth for imagesbooks directory [NOT GENERIC]
        '''
        queryfilename = os.path.basename(q_path)
        # print (queryfilename)
        filenumber = int(queryfilename.split('.')[0].split('ukbench')[1])
        fileext = queryfilename.split('.')[1]
        start = filenumber - filenumber%4
        #init a list of match file names
        familyfilename = []
        
        for i in range(start, start+4):
            familyfilename.append('ukbench'+ str(i).zfill(5)+ '.' + fileext)
        # print(familyfilename)
    
        return queryfilename, familyfilename