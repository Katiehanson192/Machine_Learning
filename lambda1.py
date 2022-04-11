from tkinter import N


remainder = lambda num: num % 2 #lambda any number of expressions: format?

print(remainder(5)) 

product = lambda x,y: x*y

print(product(2,3)) #gives the product of 2*3 

#use when want to pass a function as an arguement to higher-order functions (functions that take other functinos as their arguments ex: filter(), map())

def testfunct(num):
    print(num)
    return lambda x: x*num

result10 = testfunct(10) #creates a function called result 10, num argument = 10, but don't know x value yet
result100 = testfunct(100)

print(result10(9)) #9 = value for x 
print(result100(9))

#same thing, different format
result10 = lambda x: x*10
result100 = lambda x: x*100


#filter functions
numbers_list = [2,6,8,10,11, 4,7,13,17,0,3,21]

filtered_list = list(filter(lambda num: (num > 7), numbers_list))#first arguement = function, 2nd arguement = iterable
    #goes through numbers_list and decides if it's greater than 7, if so, adds it to a list

print(filtered_list)

#map, works with filter, applies it to every element of the list and returns true/false

def addition(n):
    return n + n

numbers = [1,2,3,4]
result = map(addition, numbers)

print(list(result))

#same thing but in 2 line result
result = list(map(lambda n: n + n, numbers)) #result: [2,4,6,8]

print(result)

#sets
numbers = (1,2,3,4)
numbers2 = (5,6,7,8)

result = list(map(lambda x,y: x + y, numbers, numbers2)) #adds the 1st number of each list together, then the second numbers , etc. result = [6,8,10,12]

print(result)

#zipping
list1 = ['a', 'b','c']
list2 = [1,2,3]
list3 = [1.5,3.1, 5.7]

for item in zip(list1,list2,list3): #puts all lists together then combines them. result: item = [a, 1, 1.5, b, 2, 3.1, c, 3,5.7]
    l1, l2, l3 = item
    print(l1)
    print(l2)
    print(l3) 