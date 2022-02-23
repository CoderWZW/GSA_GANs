from averageAttack import AverageAttack
from bandwagonAttack import BandWagonAttack
from randomAttack import RandomAttack
from unorganizedMaliciousAttacks import UMAttack


def UMattack():
    attack = UMAttack('config/config.conf')
    attack.insertSpam()
    attack.generateLabels('UMlabels.txt')
    attack.generateProfiles('UMprofiles.txt')
    
def Aattack():
    attack = AverageAttack('config/config.conf')
    attack.insertSpam()
    attack.generateLabels('Alabels.txt')
    attack.generateProfiles('Aprofiles.txt')

def Rattack():
    attack = RandomAttack('config/config.conf')
    attack.insertSpam()
    attack.generateLabels('Rlabels.txt')
    attack.generateProfiles('Rprofiles.txt')

def Battack():
    attack = BandWagonAttack('config/config.conf')
    attack.insertSpam()
    attack.generateLabels('Blabels.txt')
    attack.generateProfiles('Bprofiles.txt')

if __name__ == '__main__':
    Aattack()
    Rattack()
    Battack()
    UMattack()