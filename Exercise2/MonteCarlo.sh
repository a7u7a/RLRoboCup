#!/bin/bash

# Number of defense agents must be added by one to take into account of goalkeeper
# Cannot run an environment where defending agents exist but none are playing
# goalkeeper

# "--headless" deleted after "--trials"
./../../../bin/HFO --defense-agents=2 --offense-agents=1 --offense-on-ball 11 --trials 500 --deterministic --discrete=True --frames-per-trial 2000 --untouched-time 2000 &
sleep 5
./DiscreteHFO/Initiator.py --numTrials=500 --numPlayingDefenseNPCs=1 --numAgents=1 &
echo "Environment Initialized"

# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
sleep 5

./MonteCarloBase.py --numEpisodes=10 --numOpponents=1 &
echo "Attacker Controller Initialized"

sleep 5
./DiscreteHFO/Goalkeeper.py --numEpisodes=10 &
echo "Goalkeeper Initialized"

sleep 5
./DiscreteHFO/DiscretizedDefendingPlayer.py --id=1 --numEpisodes=10 &
echo "Defending Player Initialized"

sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait