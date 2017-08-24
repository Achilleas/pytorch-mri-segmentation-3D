scriptDir="../../ANTs/"
template_fp="../../Data/Templates/mni_128x128x64.nii"
dataPath="../../Data/MS2017b/scans/*/"

for i in $(ls -d $dataPath);
do 
	FLAIR_fp=""$i"pre/FLAIR.nii.gz" && \
	WMH_fp=""$i"wmh.nii.gz" && \
	T1_fp=""$i"pre/T1.nii.gz" && \ 
	#create folder
	mkdir tempDir && \

	#copy FLAIR file to folder
	#cp $flair_fp tempDir

	$scriptDir/antsRegistrationSyNQuick.sh -d 3 -f $T1_fp -m $FLAIR_fp -o tempDir/out1 -t r && \
	$scriptDir/antsRegistrationSyNQuick.sh -d 3 -f $template_fp -m $T1_fp -o tempDir/out2 && \

	$scriptDir/antsApplyTransforms -d 3 -i $FLAIR_fp -o tempDir/FLAIR_s128.nii.gz -t tempDir/out10GenericAffine.mat -t tempDir/out20GenericAffine.mat -r tempDir/out21Warp.nii.gz -n BSpline
	$scriptDir/antsApplyTransforms -d 3 -i $WMH_fp -o tempDir/wmh_s128.nii.gz -t tempDir/out10GenericAffine.mat -t tempDir/out20GenericAffine.mat -r tempDir/out21Warp.nii.gz -n NearestNeighbor
	#out2Warp.nii.gz
	#out2GenericAffine.mat
	#out1GenericAffine.mat
	mv tempDir/FLAIR_s128.nii.gz ${i}pre/
	mv tempDir/wmh_s128.nii.gz ${i}

	rm -r tempDir
	#echo ${i};
done
#for loop
	#make directory
	#move files there
	#apply scripts
	#move created FLAIR file
	#delete directory


#"$scriptDir"/antsRegistrationSyNQuick.sh -d 3 -f 
#./antsRegistrationSyNQuick.sh -d 3 -f ../test/mni_icbm152_t1_tal_nlin_sym_09a.nii -m ../test/T1.nii.gz -o ../test/test
#./antsRegistrationSyNQuick.sh -d 3 -f ../test/T1.nii.gz -m ../test/FLAIR.nii.gz -o ../test/test2 -t r