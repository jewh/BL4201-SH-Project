#!/bin/bash
# convert-banjo-to-json.sh
# 
# Script that parses Banjo output report files and converts them into a JSON
# format which can be used as input for Multipiles.
# 
# DEBUG:set -x will run the entire script in debug mode.
#       Traces of each command plus its arguments are printed to standard 
#       output after the commands have been expanded but before they are executed.
# set -x

# Edit the $netCounter and $nodeCounter variables to match the Banjo settings file 
# before executing the script. 
# networkCounter indicates the number of top-scoring non-identical networks 
# found so far during the search.
# nodeCounter indicates the number of nodes or variables in the network.
#networkCounter=0 # TV read from report
#nodeCounter=0 # TV read from report

now=$(date +"%Y-%m-%d_%H.%M.%S")
input=$1
output=$input"-$now.json"

trap "echo The process with pid $$ got terminated by the user >> $output ; exit" SIGINT SIGTERM
echo "pid is $$"

case "$1" in
        -i)
                input=$2
                ;;
        -d)
                input=$2
                set -x
                ;;
		-h | "" | -* )
        	echo 'Usage: convert-banjo-to-json [-{hid}] [<arg>] filename'
        	echo '       -h: show help.'
        	echo '       -i: input file "<arg>"'
        	echo '       -d: debug mode'        	
        	exit
		;;
esac
	

read -p "Is $input the Banjo report file you want to convert in JSON format? (y or n) " ans
case $ans in
	y|y*|Y|Y*)
		echo "OK. Selected file: $input"
		;;
	n|n*|N|N*)
		read -p "Type the name of the file to scan: " input
		;;
	*)
		echo "..exiting script (y or n should be typed)"
		exit 1
		;;
esac	

read -p "Please enter an integer between 0 and 10 as run_id: " run_id
case $run_id in
	0|1|2|3|4|5|6|7|8|9|10)
		echo "OK. Selected run_id: $run_id"
		;;
	*)
		echo "..exiting script (integer between 0 and 10 should be typed)"
		exit 1
		;;
esac

filelist=( $(less $input))
length=${#filelist[@]}
declare -i counter=0
declare -i zp=0
out=''

echo "..reading $length items from file $input"

# START of JSON file
out='{"times": [' #> $output
echo $out

for ((i=0; i<${length}; i++));
do

	#echo ${filelist[$i]} 

	case ${filelist[$i]} in
		'variables:')
				zp=$i+1
				nodeCounter=${filelist[$zp]}
				echo 'NODES =' $nodeCounter
				;;
		'tracked:')
				zp=$i+1
				networkCounter=${filelist[$zp]}
				echo 'NETWORKS =' $networkCounter
				;;		
		'Intermediate')
				echo 'INTERMEDIATE' #>> $output
				;;
        'Statistics')
                echo 'STATS' #>> $output
				break;
                ;;  
        'Post-processing'*)
                echo 'POST' #>> $output
                ;;
        'Final')
				echo 'FINAL' #>> $output
				
			for ((i=i; i<${length}; i++));
			do	
				
                case ${filelist[$i]} in	
					'Statistics')
						echo 'STATS' #>> $output
						break #exit
						;;  	
				
					'Network')
						#TV echo " "
						#echo "Network ${filelist[$i+1]}"
						counter=$counter+1
						#zp=$i+11
						zp=$i+3
						score=${filelist[$zp]}
						score=${score%?}
						
						zp=$zp+5
						iteration=${filelist[$zp]}
						
						echo Network$counter $score $iteration
						
						#TV if [[ $counter = "1" ]];
						#TV then
						#TV 	maxScore=$score
						#TV 	echo "MAX Score is $maxScore" 
						#TV fi
						#TV score=$((score / maxScore))
						#TV echo $score 
						
						if [[ $counter = "1" ]];
						then
							out+="{\"run_id\": $run_id, \"name\": $counter, \"score\": $score, \"iteration\": $iteration,  \"matrix\": [" # >> $output
						else
							out+=", {\"run_id\": $run_id, \"name\": $counter, \"score\": $score, \"iteration\": $iteration,  \"matrix\": [" # >> $output
						fi
						
						zp=$zp+3
						
						#echo "Score for Network $counter is $score"
						# ${filelist[$score]}"
						
						#echo -n "{\"name\": $counter, \"matrix\": [" >> $output
						#TV echo Network$counter
						#TV echo ${filelist[$zp-1]} "node has" ${filelist[$zp]} "parents"; 
						
						#TV echo "The zp will be updated $nodeCounter-1 times"
						#TV echo "This number of iterations is required for filling the rows"
						for ((j=0; j<$nodeCounter; j++));
						do
							# TODO read the parents of the current node and fill the row
							# with 0 or 1.
							# This row:
							if [[ $j>0 ]]	
							then
								zp=$zp+${filelist[$zp]}+2
								#TV echo "Next zp points at value" ${filelist[$zp]}	
							fi	

							out+="[" #>> $output
							for ((k=0; k<$nodeCounter; k++));
							do
								# This cell


								if [[ ${filelist[$zp]} = "0" ]]; 
								then
									out+="0.0" #>> $output
								else
									declare -i isLinked=0
									#declare -a arr
									for ((l=0; l<${filelist[$zp]}; l++));
									do
										#isLinked = 0
										parent=${filelist[$zp+$l+1]}
										#arr=("${arr[@]}" $parent)
									#TV	echo "Parent = "$parent " and k =" $k
										#for ((k=0; k<$nodeCounter; k++));
										#do
										if [[ $parent -eq $k ]];
											then
												isLinked=1 #echo -n "1," >> $output
									#TV			echo "In k=$k there is a link"
										fi
									done # ENDOF l
								#TV	echo "isLinked = $isLinked"
									out+="$isLinked.0" #>> $output
								fi
								if [[ $k -lt $nodeCounter-1 ]]	
									then
									out+=", " #>> $output
								fi

							done # ENDOF k -> CELL 
							out+="]" #>> $output	
							if [[ $j -lt $nodeCounter-1 ]]	
								then
								out="$out, " #>> $output
							fi		
						done # ENDOF j -> ROW
						out+="]}" #>> $output
					;;
				esac
			done
		;;
	esac
done

# END of JSON file
#TV echo '], "nodes": [{"name": "var1"}, {"name": "var2"}, {"name": "var3"}, {"name": "var4"}, {"name": "var5"}, {"name": "var6"}, {"name": "var7"}, {"name": "var8"}, {"name": "var9"}, {"name": "var10"}, {"name": "var11"}, {"name": "var12"}]}' >> $output

out="$out"'], "nodes": [' #>> $output
out="$out{\"name\": \"var0\"}" #>> $output
for ((i=1; i<nodeCounter; i++));
do
	out="$out, {\"name\": \"var$i\"}" #>> $output
done
out="$out"']}' #>> $output

#echo $out

echo -n $out >> $output

echo Output stored in $output 
# echo Please edit $output to remove the 2 redundant commas at the end.
