# Create a function that recieves a list (e.g. [10, 20, 30]),
# and then returns the new list in which each elements are doubled.

def list_double(list_a):
    double_list = []
    for i in list_a:
        double_list.append(i*2)
    return double_list

numbers = [10,20,30]

print(list_double(numbers))

