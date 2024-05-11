import sys
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm, poisson, uniform, binom
import matplotlib.pyplot as plt
from prob import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication

class window(QtWidgets.QMainWindow):
    def __init__(self):
        super(window, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.data = pd.read_csv('Clean Used Car Dataset.csv')

        # updating the comboBox for PieChart
        qualitative_var = (["transmission", "registration_year", "insurance_validity", "fuel_type", "ownership","car_name", "manufacturing_year"])
        for item in qualitative_var:
            self.ui.comboPie.addItem(item)
        # button click response
        self.ui.SearchPie.clicked.connect(self.get_pieChart)

        # updating the comboBox for Probability Distribution
        distribution_var = (["Normal Distribution", "Poisson Distribution", "Binomial Distribution", "Uniform Distribution"])
        for item in distribution_var:
            self.ui.comboPie_distribution.addItem(item)
        # button click response
        self.ui.SearchPie_distribution.clicked.connect(self.get_distribution)

        # updating the comboBox for 5 summary statictics
        quantitative_var = (["torque(Nm)", "kms_driven", "mileage(kmpl)", "engine(cc)", "max_power(bhp)", "seats", "price(in lakhs)"])
        for item in quantitative_var:
            self.ui.comboPie_EDA.addItem(item)
        # button click response
        self.ui.SearchPie_EDA.clicked.connect(self.get_eda)

        # updating the comboBox for Regression modelling
        regression_var = (["Predict milage", "Predict model year"])
        for item in regression_var:
            self.ui.comboPie_Regression.addItem(item)
        # button click response
        self.ui.SearchPie_Regression.clicked.connect(self.get_regression)

        # updating the comboBox for Regression modelling
        regression_var = (["Predict milage", "Predict model year"])
        for item in regression_var:
            self.ui.comboPie_Regression_predict.addItem(item)
        # button click response
        self.ui.SearchPie_Regression_predict.clicked.connect(self.get_regression_predict)

        # button click response for confidence interval
        self.ui.SearchPie_Confidence.clicked.connect(self.get_Confidence)


    #Function for pie chart button click
    def get_pieChart(self):
            self.ui.label_5.setText(".") # clearing the screen text
            type = self.ui.comboPie.currentText() # getting the value from comboBox
            value_counts = self.data[type].value_counts() # Calculate value counts for the column
            # Selecting top categories and grouping the rest into "Other"
            top_categories = value_counts.head(10)  
            other_sum = value_counts[10:].sum()  

            if other_sum != 0 :
              top_categories['Other'] = other_sum

            # Calculate percentages
            total = top_categories.sum()
            percentages = (top_categories / total * 100).round(1)

            # Create labels with percentages
            labels = [f'{name}: {percent}%' for name, percent in zip(top_categories.index, percentages)]

            # Create a pie chart
            plt.figure(figsize=(13, 6))
            plt.pie(top_categories, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(range(len(top_categories))))
            plt.title(f'Pie Chart of {type} with Percentages')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

            # Calculate the mode
            mode_value = self.data[type].mode()[0]   

            # description contains all the output text
            description = (
                f"Detailed Percentages for Each Category in {type}:\n" +
                "\n".join([f'{name}: {percent}%' for name, percent in zip(top_categories.index, percentages)]) +
                f"\n\nThe mode for {type} is: {mode_value}\nWhich means that in {type}\n {mode_value} is most popular"
            )
            # updating the text on screen
            self.ui.label_5.setText(description)

    #Function for probability Distribution button click
    def get_distribution(self):
        self.ui.label_5.setText(".") # clearing the screen text
        type = self.ui.comboPie_distribution.currentText() # getting the value from comboBox
        type = "Normal Distribution"

        if type == "Binomial Distribution":
            import matplotlib.pyplot as plt
            n = len(self.data)  # number of trials
            p = np.mean(self.data['seats'] > 5)  # probability of success
            # Generate binomial distribution data for plotting

            if 0 < p < 1:
                binomial_dist = binom(n=n, p=p)
                x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))
                plt.plot(x, binomial_dist.pmf(x), 'bo', ms=8, label='Binomial PMF')
                plt.title('Binomial Distribution for number of seats')
                plt.xlabel('Number of successes')
                plt.ylabel('Probability Mass Function (PMF)')
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.show()
            else:
                print("Probability 'p' must be between 0 and 1 but got:", p)
            
        elif type == "Poisson Distribution":
            import matplotlib.pyplot as plt
            data_sample = self.data['kms_driven'] / 1000  # Scaling down for demonstration
            # Fit a Poisson distribution to the data
            rate = np.mean(data_sample)  # The rate parameter (lambda) for Poisson
            # Generate Poisson distribution data for plotting
            poisson_dist = poisson(rate)
            x = np.arange(poisson.ppf(0.01, rate), poisson.ppf(0.99, rate))
            plt.plot(x, poisson_dist.pmf(x), 'bo', ms=8, label='poisson pmf')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()
            
        elif type == "Normal Distribution":
            import seaborn as sns
            import matplotlib.pyplot as plt
            data_sample = self.data['price(in lakhs)']
            # Fit a normal distribution to the data
            mu, std = norm.fit(data_sample)
            sns.histplot(data_sample, kde=False, color='blue', stat="density")
            xmin, xmax = plt.xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            plt.plot(x, p, 'k', linewidth=2)
            title = "Fit results: mean = %.2f,  std = %.2f" % (mu, std)
            plt.title(title)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

        elif type == "Uniform Distribution":
            import matplotlib.pyplot as plt
            data_sample = self.data['kms_driven'] / 1000  # Scaling down for demonstration
            # Fit a uniform distribution to the data
            min_val = min(data_sample)
            max_val = max(data_sample)
            width = max_val - min_val
            # Generate uniform distribution data for plotting
            uniform_dist = uniform(loc=min_val, scale=width)
            x = np.linspace(min_val, max_val, 100)
            plt.plot(x, uniform_dist.pdf(x), 'r-', lw=5, alpha=0.6, label='uniform pdf')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.show()

    #Function for SearchPie_EDA button click
    def get_eda(self):
        self.ui.label_5.setText(".") # clearing the screen text
        var = self.ui.comboPie_EDA.currentText() # getting the value from comboBox
        stats_dict = {} # dictionary that stores all of the results

        # Calculating descriptive statistics
        min_val = self.data[var].min()
        max_val = self.data[var].max()
        range_val = max_val - min_val
        quartiles = self.data[var].quantile([0.25, 0.5, 0.75])
        mode = self.data[var].mode()[0]
        iqr = quartiles[0.75] - quartiles[0.25]
        variance = self.data[var].var()
        std_dev = self.data[var].std()

        # Storing results in dictionary
        stats_dict[var] = {
            'Min': min_val,
            'Max': max_val,
            'Range': range_val,
            'Q1': quartiles[0.25],
            'Q2 (Median)': quartiles[0.5],
            'Q3': quartiles[0.75],
            'Mode' : mode,
            'IQR': iqr,
            'Variance': variance,
            'Standard Deviation': std_dev
        }
        
        # Generating box plot
        plt.figure(figsize=(10, 5))
        sns.boxplot(x=self.data[var])
        plt.title(f'Box Plot of {var}')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

        # Printing the statistical results
        for var, stats in stats_dict.items():
            # description contains all the output text
            description = (
                            f"\nStats for {var}:" +
                            "\n".join([f"{key}: {value}" for key, value in stats.items()]) +
                            "\n\nINTERPRETATION:"+
                            f"\nMeaning 25% of the values are below \n  {stats_dict[var]['Q1']} for {var}."+
                            f"\nMeaning Half the values are below \n    {stats_dict[var]['Q2 (Median)']}, and half are above for {var}."+
                            f"\nMeaning 75% of the values fall below \n     {stats_dict[var]['Q3']} for {var}."
)
            # updating the text on screen
            self.ui.label_5.setText(description)


    def get_regression(self):
        self.ui.label_5.setText(".") # clearing the screen text
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        type = self.ui.comboPie_Regression.currentText()

        if type == "Predict milage":
            X_mileage = self.data[['price(in lakhs)']]
            y_mileage = self.data['kms_driven']
            X_train_m, X_test_m, y_train_m, y_test_m= train_test_split(X_mileage, y_mileage, test_size=0.2, random_state=42)
            model_mileage = LinearRegression()
            model_mileage.fit(X_train_m, y_train_m)

            # Prediction points
            price_points = np.linspace(self.data['price(in lakhs)'].min(), self.data['price(in lakhs)'].max(), 100).reshape(-1, 1)

            # Predictions for plots
            predicted_mileage_plot = model_mileage.predict(price_points)

            # Plotting Mileage Predictions
            plt.figure(figsize=(12, 6))
            sns.scatterplot(x=X_mileage.squeeze(), y=y_mileage, color='blue', alpha=0.5)
            plt.plot(price_points, predicted_mileage_plot, color='red', linewidth=2)
            plt.title('Predicted Mileage vs. Price')
            plt.xlabel('Price (in lakhs)')
            plt.ylabel('Mileage (kms)')
            plt.show()

        elif type == "Predict model year":
            X_year = self.data[['price(in lakhs)']]
            y_year = self.data['manufacturing_year']
            X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_year, y_year, test_size=0.2, random_state=42)
            model_year = LinearRegression()
            model_year.fit(X_train_y, y_train_y)

            # Prediction points (can be expanded or adjusted)
            price_points = np.linspace(self.data['price(in lakhs)'].min(), self.data['price(in lakhs)'].max(), 100).reshape(-1, 1)

            predicted_year_plot = model_year.predict(price_points)

            # Plotting Model Year Predictions
            sns.scatterplot(x=X_year.squeeze(), y=y_year, color='green', alpha=0.5)
            plt.plot(price_points, predicted_year_plot, color='red', linewidth=2)
            plt.title('Predicted Model Year vs. Price')
            plt.xlabel('Price (in lakhs)')
            plt.ylabel('Model Year')
            plt.show()


    def get_regression_predict(self):
        self.ui.label_5.setText(".") # claering the screen output
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        type = self.ui.comboPie_Regression_predict.currentText()
        price_in_lakhs = self.ui.spinBox_predict.value()
        # Predicting Mileage based on Price
        X_mileage = self.data[['price(in lakhs)']]  # Independent variable
        y_mileage = self.data['kms_driven']         # Dependent variable

        # Splitting the data for the mileage model
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_mileage, y_mileage, test_size=0.2, random_state=42)

        # Create and train the mileage model
        model_mileage = LinearRegression()
        model_mileage.fit(X_train_m, y_train_m)

        # Predicting Model Year based on Price
        X_year = self.data[['price(in lakhs)']]  # Independent variable
        y_year = self.data['manufacturing_year']  # Dependent variable

        # Splitting the data for the model year model
        X_train_y, X_test_y, y_train_y, y_test_y = train_test_split(X_year, y_year, test_size=0.2, random_state=42)

        # Create and train the model year model
        model_year = LinearRegression()
        model_year.fit(X_train_y, y_train_y)

        # Use the models to predict based on the input price
        price_point = [[price_in_lakhs]]

        predicted_mileage = model_mileage.predict(price_point)
        predicted_year = model_year.predict(price_point)

        if type == "Predict milage":
            description = (f"Predicted Mileage for price Rs:{price_in_lakhs} lakhs: {predicted_mileage[0]:.2f} km")
            self.ui.label_5.setText(description)

        elif type == "Predict model year":
            description = (f"Predicted Model Year for price Rs:{price_in_lakhs} lakhs: {int(predicted_year[0])}")
            self.ui.label_5.setText(description)


    def get_Confidence(self):
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        self.ui.label_5.setText(".")
        # Fit a linear regression model using ols
        model = ols('Q("price(in lakhs)") ~ Q("kms_driven")', data=self.data).fit()
        print(model.summary())

        new_data = pd.DataFrame({'kms_driven': [50000]})
        new_data_with_const = sm.add_constant(new_data, has_constant='add')

        # Predict using the model
        predicted_price = model.predict(new_data_with_const)

        # Calculate the prediction interval
        _, lower, upper = wls_prediction_std(model, exog=new_data_with_const, alpha=0.05)

        description = (f"Predicted Price (in lakhs) for 50,000 kms driven:\n     {predicted_price.iloc[0]}" + 
                       f"\n\n95% prediction interval for a car with 50,000 kms driven: \n    {lower[0]} to {upper[0]} lakhs" + 
                       f"\n\nWhich means that the test is accepted as \n     {lower[0]} < {predicted_price.iloc[0]} < {upper[0]}")
        print(description)
        self.ui.label_5.setText(description)

def main():
    app = QApplication(sys.argv)
    ex = window()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()