

# Import required libraries
import matplotlib.gridspec as gs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
This project focuses on analyzing military expenditure and human casualties across different regions from 1960 to 2021. 
To achieve a comprehensive analysis, multiple datasets will be utilized. 
All necessary datasets will be read in for the subsequent analysis.
"""

# Loading the dataset on military expenditure
# This dataset contains information on military spending for various countries over the years.
exp = pd.read_csv('API_MS.MIL.XPND.CD_DS2_en_csv_v2_5363327.csv', skiprows=4)


# Loading the dataset on conflict-related deaths
# This dataset includes data on the number of deaths caused by state-based conflicts, categorized by world region.
death = pd.read_csv('deaths-in-state-based-conflicts-by-world-region.csv')


# Loading the dataset on the history of conflicts
# This dataset provides historical information on various conflicts and wars, detailing their occurrences and impacts.
history = pd.read_csv('conflicts and wars - Sheet1.csv')


"""
We will begin our analysis with the military expenditure dataset. 
The initial steps involve exploring the data to understand its structure and contents. 
This includes performing data wrangling operations to clean and prepare the data, 
making it suitable for further analysis.
"""

#  Displaying the structure and essential information about the military expenditure dataset
exp.info()

# Removing columns that contain categorical data and are not relevant for our analysis
# Specifically, the 'Country Code', 'Indicator Name', and 'Indicator Code' columns will be dropped
exp.drop(columns=['Country Code', 'Indicator Name',
         'Indicator Code'], inplace=True)

# Generating and displaying descriptive statistics for the dataset
# This summary includes measures such as mean, standard deviation, and quartiles
exp.describe()

# Identifying missing values within the dataset
# This step helps in understanding the extent and distribution of missing data
exp.isna().sum() # there are missing values in our data

# Addressing the missing data by replacing all missing values with zeros
# This ensures that subsequent analyses are not affected by the presence of missing values
exp.fillna(0, inplace=True)  # Filling all missing data with 0 values

"""
Given our focus on regional analysis, the dataset will be filtered to include
only regions as defined by the World Bank. This will ensure that our analysis 
is consistent and comparable across recognized regional groupings.
"""

# Filtering the dataset to include only regions as defined by the World Bank
# This step is essential to focus our analysis on recognized regional groupings
fil = exp['Country Name'].isin(['East Asia & Pacific', 'Europe & Central Asia', 'Latin America & Caribbean',
                                'Middle East & North Africa', 'South Asia', 'Sub-Saharan Africa', 'North America'])

# Applying the filter to keep only the rows corresponding to the specified regions
exp = exp[fil]  
exp.reset_index(drop=True, inplace=True)  # reseting the index


# Analyzing the trend of military expenditure across the years for each region
# Transposing the dataset to switch rows and columns for easier manipulation
exp_year = exp.T  
exp_year.columns = exp_year.iloc[0]  # Making the first row the column name
exp_year = exp_year.iloc[1:]  # filtering only for years
# changing the datatype
exp_year = exp_year.apply(pd.to_numeric, errors='coerce')

# Calculating total expenditure for each yeqar
exp_year['Total'] = exp_year.sum(axis=1)
exp_year = exp_year.reset_index()  # reset the index
exp_year.rename(columns={'index': 'Year'}, inplace=True)  # renaming columns
exp_year['Year'] = exp_year['Year'].astype(
    'float')  # changing the year to a date
exp_year.head()

# calculating total expenditure for each region
exp['Total'] = exp.sum(axis=1)
exp_region = exp.sort_values('Total')  # sorting the data in descending order
exp_region.head()


"""
Next, we will explore and analyze the dataset related to conflict deaths. 
This will help us understand the extent of human casualties resulting
 from various conflicts across different regions.
"""

# Displaying basic information about the conflict deaths dataset
death.info()

# # Generating summary statistics to get an overview of the dataset
death.describe()

# Identifying the different regions included in the dataset
death.isna().sum()  # No missing data

# Identifying the different regions included in the dataset
death['Entity'].value_counts()

# renaming the column
death.rename(columns={
      'Deaths in all state-based conflict types': 'Total Death'}, inplace=True)


# Filtering the dataset to include only the specified regions
filt = death['Entity'].isin(
    ['Africa', 'Americas', 'Asia & Oceania', 'Europe', 'Middle East'])

death_region = death[filt]

# Grouping the data by region and summing up the total deaths
death_region = death_region.groupby('Entity')['Total Death'].sum()

death_region = pd.DataFrame(death_region)  # converting to a dataframe format

death_region


"""
Finally, we will explore the dataset on the history of conflicts to analyze 
the regions where conflicts have occurred most frequently from 1960 to the present.
"""

# Displaying the first few rows of the conflict history dataset
history.head()

# Displaying basic information about the conflict history dataset
history.info()

# checking if there are missing values in our data
# we have no missing data in the columns we want to use for our analyzing
history.isna().sum()

# filtering the data from years 1960-2022
year = (history['Date'] >= '1960') & (history['Date'] <= '2022')
history = history[year]
history

# checking the  regions  columns to ensure no duplicates
history['Region'].value_counts()

# Renaming and correcting misspelled region names for consistency
history['Region'] = history['Region'].replace({'Latin America and the Ca': 'Latin America and the Caribbean',
                                              'Latin America & the Caribbean': 'Latin America and the Caribbean',
                                               '*Western Asia': 'Western Asia'})

# counting regions where conflict occurs the most
conflict_region = history['Region'].value_counts()
conflict_region.head()

# converting this to a Dataframe
conflict_region = pd.DataFrame(conflict_region)
conflict_region = conflict_region.reset_index()  # resetting the index
conflict_region.head()

# renaming the columns
conflict_region.columns = ['Region', 'Total Number']
# filtering out regions with no/insignificant occurence of war
conflict_region = conflict_region.head(10)
conflict_region.head



"""
We will now create a dashboard to visualize various aspects of the datasets,
including trends in military expenditure, total expenditures by region, 
the number of conflicts per region, and the distribution of deaths by region. 
This will be achieved through a combination of line plots, bar charts, and pie charts.
"""



# Setting the style
plt.style.use('default')

# creating a figure
fig = plt.figure(figsize=(11, 11), dpi=300)

# creating a gridspec object
gs = gs.GridSpec(4, 4, wspace=0.8, hspace=1.3)

ax1 = plt.subplot(gs[0:2, 1:4])  # line plot
ax1.plot(exp_year['Year'], exp_year['Sub-Saharan Africa'],
         label='Sub-Saharan Africa')
ax1.plot(exp_year['Year'], exp_year['South Asia'], label='South Asia')
ax1.plot(exp_year['Year'], exp_year['North America'], label='North America')
ax1.plot(exp_year['Year'], exp_year['Middle East & North Africa'],
         label='Middle East & North Africa')
ax1.plot(exp_year['Year'], exp_year['Latin America & Caribbean'],
         label='Latin America & Caribbean')
ax1.plot(exp_year['Year'], exp_year['Europe & Central Asia'],
         label='Europe & Central Asia')
ax1.plot(exp_year['Year'], exp_year['East Asia & Pacific'],
         label='East Asia & Pacific')
ax1.legend(fontsize=8)
ax1.set_title('Trend of Region Military Expenditure (1960-2021)',
              size=14, weight=1000)
ax1.set_xlabel('Year', weight=1000, size=12)
ax1.set_ylabel('Expenditure (in Billions USD)', weight=1000, size=12)

# plotting a bar chart to show total expenditure for each region
ax2 = plt.subplot(gs[0:2, 0:1])  # pie chart
ax2.barh(exp_region['Country Name'], exp_region['Total'], color='r')
ax2.set_title('Military Expenditure by Region',
              size=13, weight=1000)
ax2.set_xlabel('Expenditure (in Billions USD)', size=12, weight=1000)
ax2.set_ylabel('Region', size=14, weight=1000)

# plotting a bar chart to visualize a number of occurrences in regions
ax3 = plt.subplot(gs[2:4, 0:2])  # bar plot
ax3.barh(conflict_region['Region'], conflict_region['Total Number'])
plt.gca().invert_yaxis()
ax3.set_title('Regions With Most Conflicts', size=14, weight=1000)
ax3.set_xlabel('Number of Conflicts', size=12, weight=1000)
ax3.set_ylabel('Region', size=12, weight=1000)

# Plotting a pie chart to show proportion of total death by region
ax4 = plt.subplot(gs[2:4, 2:4])  # pie chart
ax4.pie(death_region['Total Death'], labels=death_region.index)
ax4.set_title('Distribution of Death by Regions', size=14, weight=1000)
ax4.set_xlabel('Proportion of Total Death', size=14, weight=1000)


# create a text box with summary of the our dashboard and analysis
summary_text = (
    "Summary Report\n\n"
    "This project analyzed military spending and human casualties from 1960 to 2021 across different regions of the world.\n\n"
    "North America leads in military expenditure with over $2 trillion USD, about 40% of the global total. Europe and East Asia\n\n"
    "each account for 20% of global military spending. Western Asia has the highest conflict frequency, about 25% of all conflicts,\n\n"
    "followed by Sub-Saharan Africa and Southern Asia at 15% each. Asia & Oceania suffer the most conflict-related deaths, 40% \n\n"
    "of the total, followed by Africa at 30% and the Middle East at 20%. Despite high military spending, North America has fewer \n\n"
    "conflicts has fewer conflicts and casualties, while regions with lower spending face higher human losses. Higher military  \n\n"
    "investments may lead to better defense mechanisms and fewer casualties."
    
)

textbox = plt.text(0.5, -0.17, summary_text, transform=fig.transFigure,
                   fontsize=15, fontweight='bold', horizontalalignment='center')

plt.subplots_adjust(bottom=0.2)


# Adding a subtitle to the Dashboard
plt.suptitle('Global Analysis of Military Expenditure and Conflict Casualties (1960-2021)\n\nBy Reginald Charles-Granville \n\n(Student Number: 22013636)',
             weight=1000, size=19, y=1.08)
plt.subplots_adjust(top=0.9)


# Setting a boarderline around the dashboard
fig = plt.gcf()
fig.patch.set_linewidth('5')  # set the width of the figure border
fig.patch.set_edgecolor('black')  # set the color of the figure border


# Saving the dashboard as a PNG file
plt.savefig('22013636.png', dpi=300)

plt.show()
