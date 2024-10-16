# Create a Car class with the following specifications:
# Attributes:
# make: Make of the car (string)
# model: Model of the car (string)
# year: Year the car was manufactured (integer)
# Methods:
# __init__: Constructor method to initialize the attributes.
# display_info: Method that prints out the make, model, and year of the car.
# Create an instance of the Car class, set some values for its attributes, and display its information.

class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year
    def display_info(self):
        print(f"Car make: {self.make}")
        print(f"Car model: {self.model}")
        print(f"Car year: {self.year}")

c1 = Car("Mazda", "RX-7", 1995)
c1.display_info()