%%Christian Hansen
%%Matlab R2010a
%%Wednesday, August 26th, 2015

%%This code will explore several parameters effects upon the 3-year default
%%rate, by plotting these values against the 3-year default rate.


clear
M = importdata('sixup.dat');
%%import data as .dat so that I can use it properly and remove the grand
%%total from the bottom of the csv made from the pivot table file so not to misrepresent the data 
%%look at data from the .dat. used max values because average brought up
%%division by zero.

%%list of column names as stored in the .dat file and .csv
colnames = {'Max of Federal Loan 3-Year Default Rate';...
    'Max of 2012 4-Year Grad Rate';...
    'Max of % Pell Recipients Among Freshmen';...
    'Max of Estimated Median SAT / ACT';...
    'Max of Average Student Loan (all sources)';...
    'Max of Average Net Price After Grants';...
    'Max of % Degrees Awarded in Science, Technology, Engineering, and Math';...
    'Max of % Admitted';...
    'Max of Average High School GPA Among College Freshmen'};

%%getting rid of zeros which were what was from the original
%% data set marked either non-value (-) or ds. The pivot table
%% automatically changed these values.
N= 2229; % row count

input_value = input('which values to plot (2-9): ');

figure();
for i=1:2229
    %%didn't work as expected, but the range can be adjusted manually to
    %%see trends, given further time I'd spend my efforts working out the
    %%filtering as that should be a key step.
    
if M(i,1) > 0 && M(i,input_value) > 0
     m1(i) = M(i,1);
     m2(i) = M(i,input_value);
% else
%     m1(i)=10;
%     m2(i) = 10;
end
end

%%I tried to do fitting with regression, but ended up 
%%simply using the basic fitting capabilities.

plot(m2,m1,'k.')
xlabel(colnames(input_value))
ylabel(colnames(1))
lab1=char(colnames(1));
lab2=char(colnames(input_value));

str=sprintf('%s versus %s',lab1,lab2);
title(str)