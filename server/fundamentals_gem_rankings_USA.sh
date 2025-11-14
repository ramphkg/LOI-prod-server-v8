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

echo "USA"
echo "1) Fetch(prerun - upsert fundamentals  data)"
python fundamentals_fetcher.pyc --source FINNHUB --country USA  --verbose
sleep 5
echo "2) Gnerate Ranks and resfresh Watchlists"
python fundamentals_ranker.pyc --source FINNHUB --country USA  --verbose
sleep 5

echo "Pipeline finished."
