import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# loading ML model
with open("ECG.pkl", 'rb') as f:
    ecg = pickle.load(f)
heart_diseases = {0:'Normal beat',
                  1:'Supraventricular premature beat',
                  2:'Premature ventricular contraction',
                  3:"Fusion of ventricular and normal beat",
                  4:'Unclassifiable beat'
                 }

info = {1:"""INTRODUCTION

Supraventricular premature beats represent premature activation of the atria from a site other than the sinus node and can originate from the atria (premature atrial complexes [PACs]; also referred to as premature atrial beats, premature supraventricular complexes, or premature supraventricular beats) or the atrioventricular node (called junctional premature beats [JPBs]), though the vast majority are atrial in origin. PACs are triggered from the atrial myocardium in a variety of situations and occur in a broad spectrum of the population. This includes patients without structural heart disease and those with any form of cardiac disease, independent of severity.
The prevalence, mechanisms, clinical manifestations, diagnosis, and treatment of PACs will be presented here. A discussion of premature ventricular complex/contraction (PVC; also referred to a premature ventricular beats or premature ventricular depolarizations) is presented separately. (See "Premature ventricular complexes: Clinical presentation and diagnostic evaluation".)
PREVALENCE

PACs are fairly ubiquitous, occurring commonly in both young and older adult subjects and in those with and without significant heart disease.

The prevalence of PACs is highly dependent upon the technique used for evaluation. PACs are less commonly seen on standard 10-second electrocardiogram (ECG) compared with 24-hour or longer duration Holter monitoring. In a cross-sectional analysis of 1742 Swiss adults (50 years of age or older) from the general population who underwent Holter monitoring for 24 hours, 99 percent had at least one PAC during the monitoring period [1]. In this Swiss cohort, the frequency of PACs steadily increased with age, with rates of 0.8, 1.4, and 2.6 PACs per hour among participants aged 50 to 55 years, 60 to 65 years, and 70 or more years, respectively [1]. Similar findings of greater PAC frequency with advancing age have been reported in other cohorts as well [2-4].

The presence and frequency of PACs is dependent upon the presence of structural heart disease. PACs are particularly frequent in patients with mitral valve disease and in those with left ventricular dysfunction regardless of etiology. However, the high prevalence of PACs in the normal population makes such associations uncertain.""",
        2:"""Overview
Premature ventricular contractions (PVCs) are extra heartbeats that begin in one of the heart's two lower pumping chambers (ventricles). These extra beats disrupt the regular heart rhythm, sometimes causing a sensation of a fluttering or a skipped beat in the chest.

Premature ventricular contractions are a common type of irregular heartbeat (arrhythmia). premature ventricular contractions (PVCs) are also called:

Premature ventricular complexes
Ventricular premature beats
Ventricular extrasystoles
Occasional premature ventricular contractions in people without heart disease usually aren't a concern and likely don't need treatment. You might need treatment if the premature ventricular contractions are very frequent or bothersome, or if you have an underlying heart condition.

Symptoms
Premature ventricular contractions often cause few or no symptoms. But the extra beats can cause unusual sensations in the chest, such as:

Fluttering
Pounding or jumping
Skipped beats or missed beats
Increased awareness of the heartbeat
When to see a doctor
If you feel fluttering, pounding or a sensation of skipped heartbeats in your chest, talk to your health care provider. A health care provider can determine if the sensations are due to a heart condition or other health concern. Similar signs and symptoms can be caused by many other conditions such as anxiety, low red blood cell count (anemia), overactive thyroid (hyperthyroidism) and infections.
Causes
To understand the cause of premature ventricular contractions (PVCs), it might help to learn more about how the heart typically beats.

The heart is made of four chambers — two upper chambers (atria) and two lower chambers (ventricles).

The heart's rhythm is controlled by a natural pacemaker (the sinus node) in the right upper chamber (atrium). The sinus node sends electrical signals that typically start each heartbeat. These electrical signals move across the atria, causing the heart muscles to squeeze (contract) and pump blood into the ventricles.

Next, the signals arrive at a cluster of cells called the atrioventricular (AV) node, where they slow down. This slight delay allows the ventricles to fill with blood. When the electrical signals reach the ventricles, the chambers contract and pump blood to the lungs or to the rest of the body.

In a typical heart, this heart signaling process usually goes smoothly, resulting in a resting heart rate of 60 to 100 beats a minute.
Typical heartbeat
In a typical heart rhythm, a tiny cluster of cells at the sinus node sends out an electrical signal. The signal then travels through the atria to the atrioventricular (AV) node and into the ventricles, causing them to contract and pump blood.

PVCs are irregular contractions that start in the ventricles instead of the atria. The contractions usually beat sooner than the next expected heartbeat.

The cause of premature ventricular contractions isn't always clear. Certain things including heart diseases or changes in the body can make cells in the lower heart chambers electrically unstable. Heart disease or scarring may cause the heart's signals to be misrouted.

Premature ventricular contractions may be caused by:

Certain medications, including decongestants and antihistamines
Alcohol or drug misuse
Stimulants such as caffeine or tobacco
Increased levels of adrenaline in the body due to exercise or anxiety
Injury to the heart muscle due to disease
Risk factors
Certain lifestyle choices and health conditions may make a person more likely to develop premature ventricular contractions (PVCs).

Risk factors for PVCs include:

Caffeine
Tobacco
Alcohol
Stimulants such as cocaine or methamphetamines
Exercise — if you have certain types of PVCs
Anxiety
Heart attack
Heart disease, including congenital heart disease, coronary artery disease, heart failure and a weakened heart muscle (cardiomyopathy)
Complications
Having frequent premature ventricular contractions (PVCs) or certain patterns of them might increase the risk of developing irregular heart rhythms (arrhythmias) or weakening of the heart muscle (cardiomyopathy).

Rarely, when accompanied by heart disease, frequent premature contractions can lead to chaotic, dangerous heart rhythms and possibly sudden cardiac death.""",
        3:"""A fusion beat occurs when electrical impulses from different sources act upon the same region of the heart at the same time.[1] If it acts upon the ventricular chambers it is called a ventricular fusion beat, whereas colliding currents in the atrial chambers produce atrial fusion beats.

Ventricular fusion beats can occur when the heart's natural rhythm and the impulse from a pacemaker coincide to activate the same part of a ventricle at the same time, causing visible variation in configuration and height of the QRS complex of an electrocardiogram reading of the heart's activity.[2] This contrasts with the pseudofusion beat wherein the pacemaker impulse does not affect the complex of the natural beat of the heart. Pseudofusion beats are normal. Rare or isolated fusion beats caused by pacemakers are normal as well, but if they occur too frequently may reduce cardiac output and so can require adjustment of the pacemaker.""",
        4:"Non supported beat!"
       }

st.image("cropedLogo.png")
st.title("I-Care")
st.info("Electrocardiography ECG Scan - Easy Healthcare for Anyone Anytime")

uploaded_file = st.file_uploader("Choose an image...", type=["csv", "xlsx"])

if uploaded_file is not None:
    # if uploaded_file.e
    d = pd.read_csv(uploaded_file)
    if d.isnull().sum().sum() !=0:
        d.fillna(d.mean())
    d = d.iloc[0,:187].values
    # d = d.values.reshape(1,-1)
    output = int(ecg.predict(d.reshape(1,-1))[0])
    disease = heart_diseases[output]    plt.grid()
    fig, ax = plt.subplots()
    ax.plot(d, label=f"{disease}", c='firebrick')
    plt.suptitle("Electrocardiography ECG")
    plt.legend(loc='upper right')
    st.pyplot(fig)
    st.write(f"{disease} detected!")
    if outpur !=0:
        st.write(f"{info[output]}")
    
