# GBM-Classification
Modules Needed: 
samtools/1.8 (http://www.htslib.org/download/)

htslib/1.8 (http://www.htslib.org/download/)

bcftools/1.8 (http://www.htslib.org/download/)

gdc-client/1.3 (https://gdc.cancer.gov/access-data/gdc-data-transfer-tool)

Tandem Repeat Finder (https://tandem.bu.edu/trf/trf.download.html)

Also download hg38.fa (http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/)

Other libraries and languages: python, python3, g++, xgboost, sklearn, numpy.

================================================================

# To Download files from TCGA using gdc-client/1.3 

Documentation: https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/

Example command to download 1 bam file from TCGA
gdc-client download d4f752cd-159b-40a3-9bdc-bc21cfb933ed -t t.txt  
Here t.txt is a token file which is generated once you have access to TCGA database

If there are many files to be downloaded, you can generate a text file which contains the list of files to be downloaded, this is known as a manifest file (gdc_manifest_file.txt in the example below)

Example

gdc-client download -m gdc_manifest_file.txt -t t.txt 

================================================================

# Extract out the fasta sequence for each chromosome from a BAM file downloaded from TCGA and detect tandem repeats.  

Example shows extracting out fasta sequence for chromosome 1 and then the tandem repeats using benson trf 
 
samtools view -b file_name.bam chr1>chr1.bam

samtools mpileup -E -uf ~/hg38.fa chr1.bam > chr1.mpileup

bcftools call -mv -Oz chr1.mpileup>chr1.vcf.gz

tabix -p vcf chr1.vcf.gz

samtools faidx ~/hg38.fa chr1|bcftools consensus chr1.vcf.gz -o chr1.fa

chmod +x ~/trf409.legacylinux64

~/trf409.legacylinux64 chr1.fa 2 7 7 80 10 50 500 -h -d -l 6

mv chr1.fa.2.7.7.80.10.50.500.dat chr1.dat

Each row in chr1.dat stores relevant information regarding a single tandem repeat region of chromosome 1 in the sample being analyzed

================================================================
# Mutation Estimation (Tang et al.)

~/tang chr1.dat chr1_data.txt

================================================================

# Align the regions to the repeats found in hg38.fa

python PPA/ppa.py chr1_data.txt align_check_1.txt 1 (in general replace 1 by chromosome number everywhere)

Create a data file with an aligned mutation profile for chromosome 1

python readalign.py align_check_1.txt out_align_check_1.txt

out_align_check_1.txt is the aligned mutation profile for chromosome 1 for the given sample

================================================================

# Machine learning part

Once you have generated aligned mutation profiles for all the samples and for all the chromosomes, you need to store them in directories in the format dir_cancer/chrchrom_numcancer_name
For example if the cancer name is brain, then the directory could be brain/chr1brain for storing aligned mutation profiles for chrom_num = 1 for all the brain samples.

In pairwise_test.py, you can choose what chromosomes you want to analyze by changing the array ``chromosomes''. The cancertypes can be added in the code by using

TCGA_code = ("dir_cancer","cancer_name",chromosomes)

For example in order to add prostate cancer, the command would be
PRAD = ("prostate","prostate",chromosomes)

To run the pairwise classifier for generating the seriation matrix use
2020_June_cancer_XGB.py

To create cancer risk profiles using a gradient boosting based multiclassfier use
2020_June_cancer_XGB_multi.py
  



