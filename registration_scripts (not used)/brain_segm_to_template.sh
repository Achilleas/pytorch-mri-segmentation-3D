#!/bin/bash
scriptDir="../../ANTs/"
template_fp="../../Data/Templates/mni_128x128x64.nii"
dataPath="../../Data/MS2017b/gifs/*/"

outParcellation="tempDir/parcellation_s128.nii.gz"
outCircumference="tempDir/circumference_s128.nii.gz"

replace="scans"

for i in $(ls -d $dataPath);
do
	parcellation_fp=""$i"parcellation.nii.gz" && \
	circumference_fp=""$i"circumference.nii.gz" && \

	T1_fp=""$i"pre/T1.nii.gz"
	#substitude gifs with scans (that's where T1 is)
	T1_fp=${T1_fp/gifs/scans} && \
	echo $T1_fp
	
	#create folder
	mkdir tempDir && \

	#copy FLAIR file to folder
	$scriptDir/antsRegistrationSyNQuick.sh -d 3 -f $template_fp -m $T1_fp -o tempDir/out && \
	$scriptDir/antsApplyTransforms -d 3 -i $parcellation_fp -o $outParcellation -t tempDir/out0GenericAffine.mat -r tempDir/out1Warp.nii.gz -n NearestNeighbor &&\
	$scriptDir/antsApplyTransforms -d 3 -i $circumference_fp -o $outCircumference -t tempDir/out0GenericAffine.mat -r tempDir/out1Warp.nii.gz -n NearestNeighbor &&\
	exit 0
	#mv $outParcellation ${i}
	#mv $outCircumference ${i}
	#rm -r tempDir
done

#for loop
	#make directory
	#move files there
	#apply scripts
	#move created FLAIR file
	#delete directory