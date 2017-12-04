def getAllIndexFromCharacter(name):
    if name == 'nondecimal':
        return 0
    elif name == 'decimal':
        return 1

def getAllCharacterFromIndex(name):
    if name == 0:
        return 'nondecimal'
    elif name == 1:
        return 'decimal'