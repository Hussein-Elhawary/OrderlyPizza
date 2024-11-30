import json


class Order:
    """
    Represents a generic order containing pizzas and drinks.
    """
    def __init__(self, customer_name):
        """
        Initialize an order with a customer name.

        Args:
            customer_name (str): Name of the customer placing the order.
        """
        self.pizza_orders = [PizzaOrder]
        self.drink_orders = [DrinkOrder]

    def add_pizza_order(self, pizza_order):
        """Add a pizza order to the list of pizza orders."""
        self.pizza_orders.append(pizza_order)

    def add_drink_order(self, drink_order):
        """Add a drink order to the list of drink orders."""
        self.drink_orders.append(drink_order)

    def get_order(self):
        """Returns the order"""
        pass


class Topping:
    """
    Class to represent an individual topping for a pizza.
    """
    def __init__(self, name, quantity=None, negate=False):
        """
        Initialize a topping for a pizza.

        Args:
            name (str): The name of the topping (e.g., 'cheese').
            quantity (str, optional): Quantity of the topping (e.g., 'extra', 'a lot'). Defaults to None.
            negate (bool, optional): Whether the topping is negated (e.g., 'no onions'). Defaults to False.
        """
        self.name = name
        self.quantity = quantity
        self.negate = negate


class PizzaOrder:
    """
    Represents a pizza order with size, toppings, and number.
    """
    def __init__(self, size, toppings=None, number=1, style=None ,negate_style=False):
        """
        Initialize a pizza order with size, toppings, number, and style (optional).

        Args:
            size (str): The size of the pizza (e.g., 'small', 'medium', 'large').
            toppings (list of str, optional): List of toppings for the pizza (e.g., ['pepperoni', 'mushrooms']).
            number (int, optional): Number of pizzas. Defaults to 1.
            style (str, optional): Style of pizza (e.g., 'thin crust', 'deep dish'). Defaults to None.
        """
        self.size = size
        self.toppings = toppings if toppings is not None else []
        self.number = number
        self.style = style
        self.negate_style = negate_style

    def add_topping(self, topping):
        """Add a topping to the pizza order."""
        self.toppings.append(topping)

    def remove_topping(self, topping):
        """Remove a topping from the pizza order."""
        if topping in self.toppings:
            self.toppings.remove(topping)

    def order_to_json(self):
        """
        Convert the order to a JSON string.

        Returns:
            str: A JSON string representation of the order.
        """
        pass
        


class DrinkOrder:
    """
    Represents a drink order.
    """
    def __init__(self, drink_type, container_type=None, volume=None, number=1):
        """
        Initialize a drink order with type, container, volume, and number.

        Args:
            drink_type (str): Type of the drink (e.g., Coke, Sprite).
            container_type (str, optional): Type of container (e.g., bottle, can). Defaults to None.
            volume (str, optional): Volume of the drink (e.g., 20 fl oz). Defaults to None.
            number (int, optional): Number of drinks. Defaults to 1.
        """
        self.drink_type = drink_type
        self.container_type = container_type
        self.volume = volume
        self.number = number

    def order_to_json(self):
        """
        Convert the order to a JSON string.

        Returns:
            str: A JSON string representation of the order.
        """
        pass