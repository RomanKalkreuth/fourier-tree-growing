python_script=large_scale_poly_2.py
timestamp=$(date '+%d-%m-%Y_%Hh%Mm%Ss')
for alg in "canonical-ea"
do 
    for lambda in 1 500
    do
        if [ "$str" == "canonical-ea" -a $lambda -eq 1 ]; then
            continue
        fi
        for degree in 10 20 50 100
        do
            for constant in 'none' '1' 'koza-erc'
            do
                for (( instance=0; instance<5; instance++ ))
                do
                    $(nohup python $python_script --instance $instance --algorithm $alg --dirname Exp_${timestamp}/A-${alg}_L${lambda}_D${degree}_C-${constant}_$timestamp --degree $degree --constant $constant --lmbda $lambda > /dev/null &)
                done
            done
        done
    done
done
ps aux | grep $python_script | grep -v grep | wc -l
ps aux | grep $python_script | grep -v grep
