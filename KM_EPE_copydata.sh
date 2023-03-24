#/bin/bash

# Load input payload - file paths relative
for i in {001..555};
do    
    if [[ $i =~ ^(032|106|127|232|245|353|372|455|542|546|548|555)$ ]]; then
        echo $i
    else
        PATIENT="SURG-"
        PATIENT=$PATIENT${i}
        echo $PATIENT

        cp -r clara-models/dataset_KM/$PATIENT/output Mdrive_mount/MIP/Katie_Merriman/Project2Data/DilatedProstate_data2/$PATIENT 
    fi    
done

