import os
from MyPrediction import Prediction, display_and_export_html
from log_cleaner import log_cleaner

# apache_log_location = 'var/log/apache2/access.log'
apache_log_location = 'var/testdatashuffle.log'

output = log_cleaner(apache_log_location)
combo = Prediction(output)

savdia = 'var/csv/' + 'predict' + '.csv'

combo.to_csv(savdia, index=False, header=False)


file_name = os.path.basename(apache_log_location)
display_and_export_html(combo, file_name)
