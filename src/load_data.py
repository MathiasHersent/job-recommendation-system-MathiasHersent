import pandas as pd

def load_jobs():
    df = pd.read_csv('data/jobs.csv')
    
    # Removing useless columns
    df = df.drop(columns=["Uniq Id", "Crawl Timestamp"])
    
    # Removing rows containing NaN
    df.dropna(subset=[
        'Job Salary',
        'Location', 
        'Job Title', 
        'Job Experience Required', 
        'Key Skills', 
        'Role Category', 
        'Functional Area', 
        'Industry', 
        'Role'], inplace=True)
    
    # Formatting and cleaning the Key Skills field
    df['Key Skills'] = df['Key Skills'].apply(lambda x: x.replace('\n', ' ').replace('\r', '').lower())
    # Keeping only the minimal required job experience
    df['Job Experience Required'] = df['Job Experience Required'].apply(lambda x: x.split(' ')[0])
    
    return df