#!/bin/bash
control_c()
{
  echo -en "\n*** Ouch! Exiting ***\n"
  exit $?
}
trap control_c SIGINT

export loi_root_dir="/home/bbv_sgp/dev/LOI"
if [ "$USER" == "ram" ]
then
    export loi_root_dir="/home/ram/dev/LOI"
fi
source "$loi_root_dir/venv310/bin/activate"

export $(grep -v '^#' config_sh.cfg | xargs -d '\n')
cd  "$loi_root_dir/$PROJECTHOME/server"

pwd
python -V

echo "Deleting log files older than 7 days"
find ../log/ -type f -mtime +7 -delete

echo "load all prices run_all_markets_prices_parallel.pyc"
python run_all_markets_prices_parallel.pyc
sleep 5

# --- optimized ML pipeline (precompute -> train -> predict) per watchlist ---
echo "IN_ALL (EOD_LOCAL)"
python ta_signals_mc_parallel.pyc -w IN_ALL --source EOD_LOCAL --use_ml=yes
sleep 5
python tas_swing_scoring.pyc -w IN_ALL -s EOD 
sleep 5

echo "HK_ALL (EOD_LOCAL)"
python ta_signals_mc_parallel.pyc -w HK_ALL --source EOD_LOCAL --use_ml=yes
sleep 5
python tas_swing_scoring.pyc -w HK_ALL -s EOD 
sleep 5

# --- Handle US  ---
echo "US_ALL (FINNHUB_LOCAL)"
python ta_signals_mc_parallel.pyc -w US_ALL --source FINNHUB_LOCAL --use_ml=yes
sleep 5
python tas_swing_scoring.pyc -w US_ALL -s FINNHUB 
sleep 5

echo -e "\n-----------------------COMPLETED----------------------------------------------"
