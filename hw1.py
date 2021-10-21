class Portfolio():
    def __init__(self,cash=0):
        self.cash = cash
        self.stocks = {}
        self.funds = []
        self.hist = []

    def __str__(self):
        return "cash: $"+str(self.cash)

    def addCash(self,amount):
        self.cash = self.cash + amount
        self.hist.append("Added $ " + str(amount) + " cash")

    def withdrawCash(self,amount):
        self.cash = self.cash - amount
        self.hist.append("Withdrew $ " + str(amount) + " cash")

    def buyStock(self,amount,stock):
        self.cash = self.cash - amount * stock.price
        keys = self.stocks.keys()
        if stock.symbol in keys:
            self.stocks[stock.symbol] += amount
        else:
             self.stocks[stock.symbol] = amount
        self.hist.append("Bought " + str(amount) + " shares of " + stock.symbol)

    def buyMutualFund(self, amount,fund):
        self.cash = self.cash - amount
        keys = self.stocks.keys()
        if fund.symbol in keys:
            self.funds[fund.symbol] += amount
        else:
            self.funds[fund.symbol] = amount
        self.hist.append("Bought " + str(amount) + " shares of " + fund.symbol)

    def history(self):
        for i in range(len(self.hist)):
            print(self.hist[i])

    '''''
    def sellStock(self,amount,symbol):
        self.cash = self.cash + amount *
        keys = self.stocks.keys()
        if self.stocks[symbol] > amount:
            self.stocks[symbol] -= amount
        else:
            self.stocks[fund.symbol] = amount


    def sellMutualFund(self, amount,symbol):

        self.cash = self.cash + amount

        
    '''''
class Stock():
    def __init__(self,price,symbol):
        self.price = price
        self.symbol = symbol

class MutualFund():
    def __init__(self,symbol):
        self.symbol = symbol

portfolio = Portfolio() #Creates a new portfolio
portfolio.addCash(300.50) #Adds cash to the portfolio
s = Stock(20, "HFH") #Create Stock with price 20 and symbol "HFH"
portfolio.buyStock(5, s) #Buys 5 shares of stock s
mf1 = MutualFund("BRT") #Create MF with symbol "BRT"
mf2 = MutualFund("GHT") #Create MF with symbol "GHT"
portfolio.buyMutualFund(10.3, mf1) #Buys 10.3 shares of "BRT"
portfolio.buyMutualFund(2, mf2) #Buys 2 shares of "GHT"
print(portfolio) #Prints portfolio
#portfolio.sellMutualFund("BRT", 3) #Sells 3 shares of BRT
#portfolio.sellStock("HFH", 1) #Sells 1 share of HFH
portfolio.withdrawCash(50) #Removes $50
portfolio.history() #Prints a list of all transactions
#ordered by time


