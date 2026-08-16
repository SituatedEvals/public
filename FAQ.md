# SimulacraBench FAQ

## 1. Who can participate?

SimulacraBench is open to anyone, including participants from academia, international organizations, industry and independent research, except where participation is precluded by applicable sanctions or law.

## 2. Can I participate as an individual or with people from different organisations or countries?

Yes. You may participate individually or as part of a team. Team members may come from different organisations and countries. An individual may appear on at most one team.

## 3. What exactly are teams asked to predict?

Teams predict survey responses for respondents whose answers they have not observed, using the information provided by each survey instrument and dataset.

## 4. Do participants receive the underlying respondent microdata?

No. The underlying respondent-level evaluation data are not released. Teams submit code, which is run securely against the held-out data.

## 5. Why is the evaluation data not released?

The datasets contain unpublished real-world microdata provided by partner institutions and cannot be distributed publicly. Keeping the evaluation data hidden also helps provide a contamination-resistant test of how well methods generalise to genuinely unseen respondents.

## 6. What information does my model receive at inference time?

Each dataset provides a defined input schema containing the information available for prediction. The starter kit and dataset documentation specify exactly which fields are supplied to submitted models.

## 7. What exactly do I submit?

Teams submit executable code rather than prediction files. Submissions are run in a secure evaluation environment against held-out respondent data.

## 8. Can I use pretrained models, open-weight models or external data?

Yes, subject to the competition rules and relevant licences. Teams should disclose the models, data and other external resources used in their approach.

## 9. Can I use commercial APIs such as OpenAI, Anthropic or Gemini?

External API calls cannot be made from the secure evaluation environment because network access is disabled. Any permitted use of external models during development must comply with the competition rules.

## 10. How are submissions scored?

Submissions are evaluated primarily using probabilistic scoring, rewarding accurate and well-calibrated predictions. Full details of the scoring procedure and how results are aggregated across datasets are provided in the competition documentation.

## 11. How do the development and final phases differ?

During the development phase, teams can submit regularly and receive leaderboard feedback on a fixed subset of respondents. In the final phase, each team submits its final solution for evaluation on previously unseen respondents, and these scores determine the final rankings.

## 12. Is the goal to replace human survey respondents?

No. SimulacraBench is designed to test when and how accurately algorithmic predictions can augment human-labelled survey data, and where those predictions fail. Strong performance on the benchmark should not be interpreted as evidence that human respondents or traditional survey methods can be replaced.
