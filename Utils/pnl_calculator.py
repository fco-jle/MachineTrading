import numpy as np


# avg price based PnLCalculator
class OriginalPnLCalculator:
    def __init__(self):
        self.quantity = 0
        self.cost = 0.0
        self.market_value = 0.0
        self.r_pnl = 0.0
        self.average_price = 0.0

    def fill(self, pos_change, exec_price):
        n_pos = pos_change + self.quantity
        direction = np.sign(pos_change)
        prev_direction = np.sign(self.quantity)
        qty_closing = min(abs(self.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
        qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing
        new_cost = self.cost + qty_opening * exec_price
        if self.quantity != 0:
            new_cost += qty_closing * self.cost / self.quantity
            self.r_pnl += qty_closing * (self.cost / self.quantity - exec_price)
        self.quantity = n_pos
        self.cost = new_cost

    def update(self, price):
        if self.quantity != 0:
            self.average_price = self.cost / self.quantity
        else:
            self.average_price = 0
        self.market_value = self.quantity * price
        return self.market_value - self.cost


# avg price based PnLCalculator
class PnLCalculator:
    def __init__(self, contract_notional=1, tick_value=1):
        self.quantity = 0
        self.cost = 0.0
        self.market_value = 0.0
        self.r_pnl = 0.0
        self.average_price = 0.0
        self.contract_notional = contract_notional
        self.tick_value = tick_value

    def fill(self, pos_change, exec_price):
        n_pos = pos_change + self.quantity
        direction = np.sign(pos_change)
        prev_direction = np.sign(self.quantity)
        qty_closing = min(abs(self.quantity), abs(pos_change)) * direction if prev_direction != direction else 0
        qty_opening = pos_change if prev_direction == direction else pos_change - qty_closing

        ticks = exec_price*100
        new_cost = self.cost + qty_opening * ticks * self.tick_value
        if self.quantity != 0:
            new_cost += qty_closing * self.average_price * 100 * self.tick_value
            self.r_pnl += qty_closing * (self.average_price - exec_price) * 100 * self.tick_value
        self.quantity = n_pos
        self.cost = new_cost

    def update(self, price):
        if self.quantity != 0:
            self.average_price = self.cost / (self.quantity * 100 * self.tick_value)
        else:
            self.average_price = 0
        self.market_value = self.quantity * price * 100 * self.tick_value
        return self.market_value - self.cost


if __name__ == '__main__':
    from prettytable import PrettyTable

    x = PrettyTable()
    quantities = np.array([1, 1, -2, -10, -10, 10, -20, 30])
    exec_prices = np.array([100.0, 101.0, 100.60, 100.78, 100.49, 101.4, 101.24, 101.1])
    pnls = []
    print('Pos\t|\tR.P&L\t|\tU P&L\t|\tAvgPrc')
    print('-' * 55)
    pos = PnLCalculator(contract_notional=100000, tick_value=10)
    x.field_names = ["Quantity", "Realized", "Unrealized", "AveragePrice", "Total P&L"]
    pnls = []
    for (p, e) in zip(quantities, exec_prices):
        pos.fill(p, e)
        u_pnl = pos.update(e)
        pnls.append(u_pnl + pos.r_pnl)
        x.add_row([pos.quantity, round(pos.r_pnl), round(u_pnl), pos.average_price, round(pos.r_pnl+u_pnl)])

    print(x)
