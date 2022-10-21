import numpy as np

def fibonacci():
    n = 40
    f = [0,1]
    for i in range(2,n):
        f.append(f[i-1] + f[i-2])
    i = input("Which Fibonacci number do you want?")
    print("The "+i+"th Fibonacci number is " + str(f[int(i)-1]))

def dracula():
    minWordLength = 5
    minWordOccurence = 300

    f = open("Dracula.txt", "r")
    document = f.read()
    f.close()
    allWords = document.split()

    wordOccurences = {}
    for word in allWords:
        if (len(word)>=minWordLength):
            if (word in wordOccurences):
                wordOccurences[word] = wordOccurences[word] + 1
            else:
                wordOccurences[word]=1    
    
    for word in wordOccurences:
        if wordOccurences[word]>=minWordOccurence:
            print(word + ":" + str(wordOccurences[word]))

def bikes():
    f = open("day.csv", "r")
    weather = np.array([],dtype=int)
    rentals = np.array([],dtype=int)
    for line in f:
        elements = line.split(",")
        if elements[0] != "instant":
            weather = np.append(weather, int(elements[8]))
            rentals = np.append(rentals, int(elements[15]))
    f.close()
    print("Number of days: ", np.sum(weather==1))
    print("Average number of rentals: ", np.mean(rentals[weather==1]))
    
def main():  
    fibonacci()
    dracula()
    bikes()
    
main()
    