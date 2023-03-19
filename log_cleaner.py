import csv
import pandas as pd


def log_cleaner(folder_path):
    output = 'var/csv/' + 'apache' + '.csv'
    with open(folder_path, 'r') as input_file, open(output, 'w', newline='') as output_file:
        # Create a CSV writer
        writer = csv.writer(output_file)
        # Read each line from the input file
        writer.writerow(['IP', 'logs1', 'logs2', 'Referral', '', 'Device Info'])
        for line in input_file:
            # Split the line into fields
            fields = line.strip().split('"')
            # Write the fields to the output CSV file
            writer.writerow(fields)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(output, header=0, index_col=False)
    df2 = pd.DataFrame()

    ## Splitting column A
    new_cols = df['IP'].str.split('- -', expand=True)

    # add the new columns to the DataFrame
    df2 = pd.concat([df2, new_cols], axis=1)


    df2 = df2.rename(columns={0: 'IP', 1: 'Timestamp'})

    # concatenate the values of the first, second and thrid columns
    combined = df['logs1'] + ' ' + df['logs2']

    # concatenate the columns horizontally
    df2 = pd.concat([df2, combined], axis=1)

    df2 = df2.rename(columns={0: 'Logs'})

    referral = df['Referral']
    df2 = pd.concat([df2, referral], axis=1)

    deviceInfo = df['Device Info']
    df2 = pd.concat([df2, deviceInfo], axis=1)

    # save the modified DataFrame to a new CSV file
    df2.to_csv(output, index=False)
    return output
