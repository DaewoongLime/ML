import pandas as pd

# Declare DataFrame
fruits = pd.DataFrame({'Apples':[30,29,30,30,30,30], 'Bananas':[21,None,19,18,10,30]})
print(fruits, end='\n\n')
# or
# fruits = pd.DataFrame([[30,21],[29,20]], columns=['Apples', 'Bananas'], index=[0,1])
# print(fruits, end='\n\n')

# Declare Series
ingredients = pd.Series({'Flour':'4 cups', 'Milk':'1 cup', 'Eggs':'2 large', 'Spam':'1 can'}, name='Dinner')
print(ingredients, end='\n\n')
# or
ingredients = pd.Series(['4 cups', '1 cup', '2 large', '1 can'], index=['Flour', 'Milk', 'Eggs', 'Spam'], name='Dinner')
print(ingredients, end='\n\n')

# Read CSV file as DataFrame
# reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)

# shape
fruits.shape

# head; first 5 rows
fruits.head()

# Write DataFrame to a CSV file
fruits.to_csv('fruits.csv')

# Accessing a Series from a DataFrame
print(fruits.Apples)
# or
print(fruits['Apples'])

# Acessing a single element
print(fruits.Apples[0])

# For indexing, use iloc and loc
fruits.iloc[:, 0]
fruits.iloc[[0, 1], 0]
fruits.loc[0, 'Apples']

# use .set_index() to make a col the index col
fruits.set_index('Apples')
print(fruits)

# conditional search: put conditional query within []
print(fruits[fruits["Apples"] > 29])

# Assign a constant data to a column
# fruits['Apples'] = 30
# print(fruits)

# assign range
fruits['index_backwards'] = range(len(fruits), 0, -1)
print(fruits)

# .describe()
print(fruits.describe())

# .mean(), .unique(), .value_counts(), 
# value_counts() returns a Series with the counts of unique values


# .idxmax(), .idxmin()
# idxmax() returns the index of the first occurrence of the maximum value
print(fruits.Bananas.idxmax())

# map() to perform function on every value from the Series
fruits.Apples = fruits.Apples.map(lambda x : x-10)
print(fruits)

# apply() to perform on whole DataFrame
def apply_change(row):
    row.Bananas = row.Bananas - 10
    return row

print(fruits.apply(apply_change, axis='columns'))

# groupby() to group by a column
# it groups by the same values in the column and returns a DataFrameGroupBy object
# Apples col 의 값이 같은 것끼리 묶어서 mean 을 구함
print(fruits.groupby('Apples').Bananas.count())
print(fruits.groupby('Apples').head())

# agg() to perform multiple operations on a single column
print(fruits)
print(fruits.groupby(['Apples']).Bananas.agg([len, 'max', 'min']))

# it is also possible to use multiple columns (multi-index)
print(fruits.groupby(['Apples', 'Bananas']).agg([len, 'max', 'min']))

# .reset_index() to reset the index of a DataFrame
print(fruits.groupby(['Apples', 'Bananas']).agg([len, 'max', 'min']).reset_index())

# .sort_values() to sort by a column
# default is ascending order, set ascending=False for descending order
# it is also possible to sort by multiple columns
print(fruits.sort_values(by='Bananas'))

# .sort_index() to sort by index
print(fruits.sort_index())

# .count() vs .size()
# .count() counts non-null values in each column
# .size() counts the number of rows in each column
# for Series, .size() is the same as len()

# data types
# .dtypes returns the data types of each column
print(fruits.dtypes)

# .dtype returns the data type of a specific column
print(fruits.Bananas.dtype)

# .astype() to change the data type of a column
fruits.Bananas = fruits.Bananas.astype('float')
print(fruits.Bananas.dtype)

# pd.isnull() to check for null values
# it returns a DataFrame of boolean values
print(pd.isnull(fruits))
# or
print(fruits.isnull())

# to locate rows with null values, use .isnull() with []
print(fruits[pd.isnull(fruits.Bananas)])

# to fill null values, use .fillna()
fruits.Bananas = fruits.Bananas.fillna(0)
print(fruits)

# .rename() to rename columns
# it takes a dictionary with old names as keys and new names as values
fruits.rename(columns={'Apples': 'Apple Count', 'Bananas': 'Banana Count'}, inplace=True)
print(fruits)

# it can also rename index (used rarely)
fruits.rename(index={0: 'First', 1: 'Second'}, inplace=True)

# .rename_axis() to rename the index or columns
# it takes a string as an argument
fruits.rename_axis('Fruit Type', axis='rows', inplace=True)
print(fruits)

# concat() to concatenate DataFrames
# it takes a list of DataFrames and concatenates them along the specified axis
# axis=0 for rows, axis=1 for columns
fruits2 = pd.DataFrame({'Apple Count':[30,29], 'Banana Count':[21, 20]})
print(pd.concat([fruits, fruits2], axis=1))

# .join() to join DataFrames
# it takes another DataFrame and joins it to the current DataFrame
# it is similar to SQL JOIN
# it can also take a column to join on

