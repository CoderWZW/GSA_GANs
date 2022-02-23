import numpy as np

def prediction_shift(algorithm, attack):
    targets = []
    with open('./targets.txt') as f:
        content = f.readlines()
        for item in content:
            item = item.strip('\n')
            targets.append(item)

    rating1 = []
    res1 = []
    with open('./clean/'+ algorithm+'.txt') as f:
        content = f.readlines()
        for rate in content:
            rate = rate.strip('\n')
            rate = rate.split(' ')
            row = []
            row.append(rate[0])
            row.append(rate[1])
            row.append(rate[2])
            row.append(rate[3])
            rating1.append(row)
        for user,item,rate,prediction in rating1:
            if item in targets:
                res1.append(float(prediction))
    
    rating2 = []
    res2 = []
    with open('./'+ attack + '/' + algorithm +'.txt') as f:
        content = f.readlines()
        for rate in content:
            rate = rate.strip('\n')
            rate = rate.split(' ')
            row = []
            row.append(rate[0])
            row.append(rate[1])
            row.append(rate[2])
            row.append(rate[3])
            rating2.append(row)
        for user,item,rate,prediction in rating2:
            if item in targets:
                res2.append(float(prediction))
 
    result = (np.array(res2)-np.array(res1)).sum() / len(res1)
    
    print(algorithm + ' ' + attack + ' prediction shift:' +str(round(result,4)))
    return result

def prediction_hitratio(algorithm, attack):
    targets = []
    with open('./targets.txt') as f:
        content = f.readlines()
        for item in content:
            item = item.strip('\n')
            targets.append(item)

    with open('./clean/'+ algorithm+'.txt') as f:
        res = []
        content = f.readlines()
        for line in content:
            line = line.strip('\n').split(' ')
            for element in line:
                element = element.strip().split('(')
                total = 20
                hit = 0
                for i in element:
                    i = i.strip('*').strip('(').strip(')').split(',')
                    if len(i)>1 and i[0] in targets:
                        #print(i[0])
                        #print(targets)
                        hit += 1
                hitratio = hit / total
                #print(hitratio)
                res.append(hitratio)
        result = (np.array(res)).sum() / len(res)
        #print(algorithm + ' ' + 'clean' + ' ' +str(result))
        
        
    with open('./'+ attack + '/' + algorithm +'.txt') as f:
        res = []
        content = f.readlines()
        for line in content:
            line = line.strip('\n').split(' ')
            for element in line:
                element = element.strip().split('(')
                total = 20
                hit = 0
                for i in element:
                    i = i.strip('*').strip('(').strip(')').split(',')
                    if len(i)>1 and i[0] in targets:
                        #print(i[0])
                        #print(targets)
                        hit += 1
                hitratio = hit / total
                #print(hitratio)
                res.append(hitratio)
        result = (np.array(res)).sum() / len(res)
        print(algorithm + ' ' + attack + ' hit ratio:'  +str(round(result*100, 4)) + '%')

        return result
                        
def transferability(score):
    average = (np.array(score)).sum() / len(score)
    return (np.square(np.array(score)).sum() / (len(score) - 1)) / average


    

if __name__ == '__main__':
    '''
    print('average_PST', str(transferability([0.003204,0.003961,0.001891,0.001254])))
    print('random_PST', str(transferability([0.004654,0.001778,0.002490,0.002364])))
    print('bandwagon_PST', str(transferability([0.003644,0.000836,0.001548,0.002654])))
    print('unorganized_PST', str(transferability([0.002486,0.001664,0.001987,0.001478])))
    '''

    average = []
    random = []
    bandwagon = []
    unorganized = []
    average.append(prediction_shift('BasicMF', 'average'))
    random.append(prediction_shift('BasicMF', 'random'))
    bandwagon.append(prediction_shift('BasicMF', 'bandwagon'))
    unorganized.append(prediction_shift('BasicMF', 'unorganized'))

    average.append(prediction_shift('PMF', 'average'))
    random.append(prediction_shift('PMF', 'random'))
    bandwagon.append(prediction_shift('PMF', 'bandwagon'))
    unorganized.append(prediction_shift('PMF', 'unorganized'))

    average.append(prediction_shift('EE', 'average'))
    random.append(prediction_shift('EE', 'random'))
    bandwagon.append(prediction_shift('EE', 'bandwagon'))
    unorganized.append(prediction_shift('EE', 'unorganized'))

    average.append(prediction_shift('SVD', 'average'))
    random.append(prediction_shift('SVD', 'random'))
    bandwagon.append(prediction_shift('SVD', 'bandwagon'))
    unorganized.append(prediction_shift('SVD', 'unorganized'))

    print('average_PST', str(transferability(average)))
    print('random_PST', str(transferability(random)))
    print('bandwagon_PST', str(transferability(bandwagon)))
    print('unorganized_PST', str(transferability(unorganized)))

    average = []
    random = []
    bandwagon = []
    unorganized = []

    average.append(prediction_hitratio('BPR', 'average'))
    random.append(prediction_hitratio('BPR', 'random'))
    bandwagon.append(prediction_hitratio('BPR', 'bandwagon'))
    unorganized.append(prediction_hitratio('BPR', 'unorganized'))

    average.append(prediction_hitratio('WRMF', 'average'))
    random.append(prediction_hitratio('WRMF', 'random'))
    bandwagon.append(prediction_hitratio('WRMF', 'bandwagon'))
    unorganized.append(prediction_hitratio('WRMF', 'unorganized'))

    average.append(prediction_hitratio('APR', 'average'))
    random.append(prediction_hitratio('APR', 'random'))
    bandwagon.append(prediction_hitratio('APR', 'bandwagon'))
    unorganized.append(prediction_hitratio('APR', 'unorganized'))

    average.append(prediction_hitratio('NeuMF', 'average'))
    random.append(prediction_hitratio('NeuMF', 'random'))
    bandwagon.append(prediction_hitratio('NeuMF', 'bandwagon'))
    unorganized.append(prediction_hitratio('NeuMF', 'unorganized'))
    
    print('average_HRT', str(transferability(average)))
    print('random_HRT', str(transferability(random)))
    print('bandwagon_HRT', str(transferability(bandwagon)))
    print('unorganized_HRT', str(transferability(unorganized)))

    '''
    gan = []
    gan.append(prediction_shift('BasicMF', 'gan'))
    gan.append(prediction_shift('PMF', 'gan'))
    gan.append(prediction_shift('EE', 'gan'))
    gan.append(prediction_shift('SVD', 'gan'))

    print('gan_PST', str(transferability(gan)))

    gan = []
    gan.append(prediction_hitratio('BPR', 'gan'))
    gan.append(prediction_hitratio('WRMF', 'gan'))
    gan.append(prediction_hitratio('APR', 'gan'))
    gan.append(prediction_hitratio('NeuMF', 'gan'))
    print('gan_HRT', str(transferability(gan)))
    '''