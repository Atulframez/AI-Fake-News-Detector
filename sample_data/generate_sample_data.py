"""
Generate a realistic sample dataset for training the Fake News Detector.

This creates a balanced dataset of real and fake news snippets
so users can train and demo the model without needing external data.
"""

import csv
import os
import random

# ── Real News Samples ──────────────────────────────────────────────
REAL_NEWS = [
    "The Federal Reserve announced a 0.25 percentage point increase in interest rates on Wednesday, citing continued inflation concerns and strong labor market data. Chair Jerome Powell emphasized the decision was data-driven.",
    "Scientists at MIT have developed a new solar cell technology that achieves 30% efficiency, marking a significant breakthrough in renewable energy research. The findings were published in Nature Energy.",
    "The World Health Organization released its annual report on global health trends, highlighting improvements in childhood vaccination rates across Southeast Asia and sub-Saharan Africa.",
    "Apple Inc. reported quarterly revenue of $94.8 billion, exceeding analyst expectations. The company attributed strong performance to iPhone 16 sales and growing services revenue.",
    "NASA's Artemis III mission successfully completed its first crewed lunar orbit, bringing astronauts closer to a planned Moon landing. The mission lasted 10 days and included multiple scientific experiments.",
    "The European Union passed new regulations on artificial intelligence, requiring transparency in AI decision-making and banning certain uses of facial recognition in public spaces.",
    "Researchers at Stanford University published findings showing that regular moderate exercise reduces the risk of dementia by 35%, based on a 20-year longitudinal study of 15,000 participants.",
    "India's GDP growth rate reached 7.2% in the third quarter, driven by strong performance in the technology and manufacturing sectors, according to the Ministry of Statistics.",
    "A new study from Oxford University confirms that electric vehicles produce 50% fewer lifecycle carbon emissions compared to traditional combustion engine vehicles, even accounting for battery production.",
    "The United Nations General Assembly voted to approve a new climate finance framework, pledging $200 billion annually to help developing nations transition to clean energy sources.",
    "Google announced the release of its latest quantum computing chip, which the company claims can perform certain calculations millions of times faster than traditional supercomputers.",
    "The International Space Station celebrated 25 years of continuous human habitation, with more than 3,000 scientific experiments conducted aboard the orbital laboratory since its launch.",
    "Germany's renewable energy sector now accounts for 52% of the country's total electricity generation, according to new data from the Federal Network Agency released this month.",
    "A clinical trial conducted across 40 hospitals showed that a new mRNA vaccine against melanoma reduced cancer recurrence by 44% when combined with existing immunotherapy treatments.",
    "Japan's central bank maintained its ultra-low interest rate policy, diverging from Western central banks that have been tightening monetary policy to combat inflation throughout the year.",
    "The Centers for Disease Control and Prevention reported a 15% decrease in flu hospitalizations this season compared to last year, attributing the decline to higher vaccination rates among adults over 65.",
    "Tesla delivered 435,000 vehicles in Q3 2025, slightly below analyst estimates of 450,000, as the company faced supply chain challenges related to battery component sourcing from Asia.",
    "Brazil's Amazon rainforest deforestation rates dropped by 22% over the past year, according to satellite data analyzed by the Brazilian National Institute for Space Research.",
    "Microsoft announced a $10 billion investment in AI infrastructure, including new data centers in five countries, to support growing demand for cloud computing and artificial intelligence services.",
    "The Supreme Court ruled 6-3 in favor of expanding broadband access requirements, mandating that telecom companies provide affordable internet service to underserved rural communities.",
    "South Korea's Samsung Electronics unveiled a new generation of memory chips that offer twice the speed and half the power consumption of previous models, targeting AI and data center applications.",
    "A peer-reviewed study in The Lancet found that air pollution is responsible for approximately 8.1 million premature deaths globally each year, surpassing previous estimates by nearly 20%.",
    "Australia's Great Barrier Reef showed signs of partial recovery in its northern sections, with coral cover increasing by 12% over two years following reduced bleaching events during cooler ocean temperatures.",
    "The Bank of England held interest rates steady at 5.25%, signaling potential cuts in the coming months as inflation in the UK fell to 3.2%, closer to its 2% target.",
    "France announced plans to build six new nuclear power plants by 2040 as part of its strategy to achieve carbon neutrality and reduce dependence on imported fossil fuels.",
    "A major cybersecurity breach at a leading US healthcare provider exposed the personal data of 11 million patients, prompting congressional calls for stricter data protection legislation.",
    "China's Chang'e-6 mission successfully returned samples from the far side of the Moon, providing scientists with unprecedented material for studying the Moon's geological history and composition.",
    "The International Monetary Fund projected global economic growth of 3.1% for the year, warning of risks from geopolitical tensions, trade fragmentation, and persistent inflation in some regions.",
    "Spotify reported reaching 600 million monthly active users, with podcast consumption growing 40% year-over-year as the platform expanded into audiobooks and live audio content.",
    "Researchers discovered a high-temperature superconductor that operates at minus 23 degrees Celsius, a significant improvement that could eventually enable room-temperature superconductivity in practical applications.",
]

# ── Fake News Samples ──────────────────────────────────────────────
FAKE_NEWS = [
    "BREAKING: Scientists confirm that 5G towers are spreading a new virus that affects brain cells. Thousands of people near cell towers have reported memory loss and headaches. The government is covering it up!",
    "EXPOSED: Secret documents reveal that the Moon landing was filmed in a Hollywood studio! Former NASA employee confesses everything on deathbed. They fooled the entire world for decades!",
    "SHOCKING: Drinking bleach mixed with lemon juice cures all types of cancer within 48 hours! Big Pharma has been hiding this simple cure for decades to protect their billion-dollar profits!",
    "ALERT: The Earth is actually flat and NASA has been faking satellite images for over 60 years! A brave whistleblower has released proof that all space photos are computer-generated!",
    "BREAKING NEWS: Vaccines contain microchips that allow the government to track your every movement! A leaked internal memo from a pharmaceutical company confirms the terrifying truth!",
    "EXPOSED: Celebrities are secretly lizard people from another dimension! Leaked photos show famous actors transforming behind the scenes. Hollywood has been infiltrated by reptilians!",
    "URGENT: Scientists discover that eating chocolate for breakfast makes you lose 30 pounds in one week without any exercise! Nutritionists worldwide are stunned by this miracle discovery!",
    "BOMBSHELL: The sun is actually cold and NASA has been lying about its temperature for centuries! A physics professor was fired for publishing this groundbreaking research that challenges everything!",
    "SHOCKING REVEAL: WiFi signals cause permanent DNA damage and lead to cancer! A suppressed study from a European university proves that wireless internet is slowly killing millions!",
    "BREAKING: A time traveler from 2089 warns that a massive asteroid will destroy Earth in exactly three months! He brought video evidence from the future that experts can't debunk!",
    "MUST READ: Airplanes are spraying mind-control chemicals in the sky! Former pilot reveals the shocking truth about chemtrails and how they're being used to manipulate the population!",
    "EXPOSED: All major elections worldwide are controlled by a single secret society! They decide every president and prime minister decades in advance. Democracy is just an illusion!",
    "UNBELIEVABLE: A man discovered that rubbing garlic on your feet cures diabetes overnight! Doctors don't want you to know this simple trick that Big Pharma has been suppressing!",
    "ALERT: The ocean is actually shrinking and world governments are hiding it! Satellite data has been manipulated for years. Coastal cities are actually gaining land, not losing it!",
    "SHOCKING: Your smartphone is recording everything you say even when turned off! Tech companies sell your private conversations to advertisers and government agencies worldwide!",
    "BREAKING: Ancient pyramids were built by aliens using anti-gravity technology! New evidence found inside the Great Pyramid proves extraterrestrial involvement in human civilization!",
    "EXPOSED: All organic food is actually worse for you than regular food! A cover-up by the organic food industry has been deceiving health-conscious consumers for over 20 years!",
    "UNBELIEVABLE: Scientists accidentally create a portal to another dimension in a laboratory! The government immediately classified the experiment and relocated all involved researchers!",
    "URGENT WARNING: Drinking water contains a secret chemical that makes people obedient and suppresses free thinking! City water treatment plants add it by government order!",
    "BOMBSHELL REPORT: Climate change is a complete hoax invented by solar panel companies to sell more products! Exposed internal emails reveal the massive fraud behind global warming!",
    "BREAKING: A teenager in his garage invents a car that runs on water but oil companies bought and destroyed the patent! He was offered $500 million to stay silent about his invention!",
    "SHOCKING: Hospital workers reveal that doctors deliberately make patients sicker to increase profits! An insider report exposes the dark truth about the medical industry worldwide!",
    "EXPOSED: Birds aren't real — they're government surveillance drones! Leaked military documents show the birds were replaced with robots during the Cold War to spy on citizens!",
    "ALERT: Reading too many books causes brain damage and early-onset Alzheimer's! A controversial new study that scientists don't want you to see proves that reading is dangerous!",
    "BREAKING NEWS: The government has been hiding a cure for aging! A secret project has kept top officials alive for over 200 years using a serum derived from a deep-sea organism!",
    "MUST SEE: Proof that the Earth's core is hollow and an advanced civilization lives inside! Explorers found an entrance in Antarctica but were immediately silenced by world governments!",
    "SHOCKING: Coffee is actually a highly addictive drug more dangerous than cocaine! Research suppressed by the coffee industry reveals the terrifying effects of caffeine on the brain!",
    "EXPOSED: All professional sports are completely scripted like wrestling! Former athletes admit that every game, match, and race outcome is predetermined by team owners and betting syndicates!",
    "UNBELIEVABLE: Sleeping less than 2 hours a night actually increases your lifespan by 40 years! A hidden study proves that sleep is overrated and doctors have been wrong all along!",
    "BREAKING: A secret underwater city has been discovered in the Pacific Ocean! The structures are over 50,000 years old and contain technology far more advanced than anything we have today!",
]


def generate_dataset(output_path: str = None, num_augmented: int = 200):
    """
    Generate a training dataset by augmenting base samples.

    Args:
        output_path: Where to save the CSV.
        num_augmented: Total number of samples per class.
    """
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), 'news_dataset.csv')

    rows = []

    # Add original samples
    for text in REAL_NEWS:
        rows.append({'text': text, 'label': 'REAL'})
    for text in FAKE_NEWS:
        rows.append({'text': text, 'label': 'FAKE'})

    # Augment by paraphrasing (simple word shuffle + prefix variations)
    fake_prefixes = [
        "BREAKING: ", "SHOCKING: ", "EXPOSED: ", "ALERT: ",
        "URGENT: ", "MUST READ: ", "YOU WON'T BELIEVE: ",
        "BOMBSHELL: ", "UNBELIEVABLE: ", "WARNING: ",
    ]

    real_prefixes = [
        "According to reports, ", "New data shows that ",
        "A recent study found that ", "Officials announced that ",
        "Research indicates that ", "Experts confirm that ",
        "Analysis reveals that ", "Sources report that ",
        "The latest findings show ", "Recent developments suggest ",
    ]

    while len([r for r in rows if r['label'] == 'FAKE']) < num_augmented:
        base = random.choice(FAKE_NEWS)
        # Randomly modify
        prefix = random.choice(fake_prefixes)
        # Slight word shuffle within sentences
        sentences = base.split('.')
        random.shuffle(sentences)
        augmented = prefix + '. '.join(s.strip() for s in sentences if s.strip())
        rows.append({'text': augmented, 'label': 'FAKE'})

    while len([r for r in rows if r['label'] == 'REAL']) < num_augmented:
        base = random.choice(REAL_NEWS)
        prefix = random.choice(real_prefixes)
        augmented = prefix + base[0].lower() + base[1:]
        rows.append({'text': augmented, 'label': 'REAL'})

    # Shuffle
    random.seed(42)
    random.shuffle(rows)

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text', 'label'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Generated {len(rows)} samples → {output_path}")
    print(f"   REAL: {len([r for r in rows if r['label'] == 'REAL'])}")
    print(f"   FAKE: {len([r for r in rows if r['label'] == 'FAKE'])}")
    return output_path


if __name__ == '__main__':
    generate_dataset()

# Dataset version: v1.2 — balanced 400-sample corpus (200 REAL + 200 FAKE)
# Augmentation: prefix variation + sentence shuffle for diversity
