class Portfolio():
    def __init__(self,cash=0):
        self.cash = cash
        self.stocks = []
        self.funds = []

    def addCash(self,amount):
        self.cash = self.cash + amount
        return self.cash

    def withdrawCash(self,amount):
        self.cash = self.cash - amount
        return self.cash

    def buyStock(self,amount,stock):
        self.cash = self.cash - amount * stock.price
        self.stocks.append([stock.symbol,amount])

    def buyMutualFund(self, amount,fund):
        self.cash = self.cash - amount
        self.funds.append([fund.symbol, amount])

    '''''
    def sellStock(self,amount,symbol):
        self.cash = self.cash + amount * stock.price
        self.stocks.remove([stock,amount])

    def sellMutualFund(self, amount,symbol):

        self.cash = self.cash + amount
        self.funds.remove([fund, amount])
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
print(portfolio.cash)
#portfolio.sellMutualFund("BRT", 3) #Sells 3 shares of BRT
#portfolio.sellStock("HFH", 1) #Sells 1 share of HFH
#portfolio.withdrawCash(50) #Removes $50
#portfolio.history() #Prints a list of all transactions
#ordered by time


