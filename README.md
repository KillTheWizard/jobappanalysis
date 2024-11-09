# Job Application Rejection Analysis: "Because 2024 Job Hunting is a Bloodsport"

### "In 2024, the job market is so brutal that we're training a model on my rejections. Here's what we came up with."

This project is a survival guide (or at least an analytical one) to navigating the ruthless 2024 job market. We’re taking rejection after rejection, analyzing what we can, and trying to predict which factors lead to the dreaded “We’ve decided to go in a different direction” email.

In a world where ghosting isn’t just for dating and “we’ll keep your resume on file” is code for “better luck next time,” let’s try to make some sense of the chaos.

## Project Overview

### Goals
1. **Predictive Modeling**: Train a model to (hopefully) predict rejection patterns, because if we’re going to be rejected, we might as well see it coming.
2. **Anomaly Detection**: Spot the outliers – those applications where something unexpected (good or bad) happened, because why not spice it up with some detective work?

### Features Analyzed
We’re taking a hard look at:
- **Company**: Encoded identifier for where dreams went to die.
- **Job Title**: The role we thought we were qualified for.
- **Phase 2**: Binary indicator of whether we made it to the second interview phase, or got a swift “no.”
- **Interval**: The time it took for the heartbreak email to arrive.
- **Application Source**: LinkedIn, Indeed, a referral, or maybe a shot in the dark.
- **Resume Version**: Different versions of the resume, just in case the “edgy” one wasn’t the right call.

## Project Structure

- **jobappanalysis.py**: The code that does all the heavy lifting:
  - Loads the data and processes it.
  - Trains a model to predict rejection (so we don’t have to guess).
  - Detects anomalies, because even in rejection, some applications stand out.
  - Saves results in `JobAppResults.xlsx`, for the record.

- **JobAppData.xlsx**: The infamous data – all the applications, hopeful follow-ups, and subsequent rejections.

- **JobAppResults.xlsx**: The output file, where you’ll find:
  - Predictions of rejection (because what’s one more heartbreak).
  - Anomalies in the data, flagged for extra intrigue.

## Dependencies

Install these libraries to get started:

```bash
pip install pandas scikit-learn openpyxl matplotlib seaborn xlwings
# jobappanalysis
