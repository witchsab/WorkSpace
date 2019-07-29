import os

def accuracy_matches(q_path, imagematches, window):
    acc = 0
    # q_path = './imagesbooks/ukbench07994.jpg'
    # print (q_path)

    queryfilename = os.path.basename(q_path)
    # print (queryfilename)

    filenumber = int(queryfilename.split('.')[0].split('ukbench')[1])

    start = filenumber - filenumber%4

    #init a list of match file names

    familyfilename = []


    for i in range(start, start+4):
        familyfilename.append('ukbench'+ str(i).zfill(5)+'.jpg')

    # print(familyfilename)

    familyfilename.remove(queryfilename)



    # imagematches = [(1.0, './imagesbooks/ukbench07994.jpg'), (0.9841861435227051, './imagesbooks/ukbench07997.jpg'), (0.9727056941785788, './imagesbooks/ukbench08000.jpg'), (0.9512209503972383, './imagesbooks/ukbench08001.jpg'), (0.9472347018503713, './imagesbooks/ukbench08048.jpg'), (0.9308509982630043, './imagesbooks/ukbench08002.jpg'), (0.9301495848422062, './imagesbooks/ukbench08062.jpg'), (0.9004718978487414, './imagesbooks/ukbench03125.jpg'), (0.8799294992301836, './imagesbooks/ukbench08051.jpg'), (0.8728566632483997, './imagesbooks/ukbench03124.jpg'), (0.8709723434067984, './imagesbooks/ukbench07993.jpg'), (0.8661491759075842, './imagesbooks/ukbench03127.jpg'), (0.8568008483230735, './imagesbooks/ukbench07996.jpg'), (0.8557715568599505, './imagesbooks/ukbench03057.jpg'), (0.8525254961024937, './imagesbooks/ukbench07992.jpg'), (0.8517494888726892, './imagesbooks/ukbench03126.jpg'), (0.8275000684980399, './imagesbooks/ukbench08060.jpg'), (0.8268087203729461, './imagesbooks/ukbench08059.jpg')]
        


    #init search list for accuracy score calculation

    searchlist = []

    for item in imagematches:
        score, path = item
        searchlist.append(os.path.basename(path))

    # print (searchlist)

    searchlist.remove(queryfilename)

    overlap = list(set(familyfilename).intersection(searchlist))


    acc = (len(overlap)/len(familyfilename)) *100



    return (acc)


# accuracy_matches ('./imagesbooks/ukbench07994.jpg', imagematches, 50 )