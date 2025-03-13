/**
 * Get step by step solution for a question
 * @customfunction
 * @param {string} question The full text of the question
 * @param {string} option_a Text for option A
 * @param {string} option_b Text for option B
 * @param {string} option_c Text for option C
 * @param {string} option_d Text for option D
 * @param {string} option_e Text for option E
 * @param {string} correct_answer_letter The letter of the correct answer (A-E)
 * @returns {string} Step-by-step solution to the problem
 */
function getStepByStep(question, option_a, option_b, option_c, option_d, option_e, correct_answer_letter) {
    prompt = `
    You are an expert in math pedagogy. Your task is to answer the following question deconstructing it to the most elemental steps. Don't skip any step. Don't assume anything about the reader, try to solve it in the most atomic and pedagogical way possible.
  
    Here is the question:
    
    Question: ${question}
    A) ${option_a}
    B) ${option_b}
    C) ${option_c}
    D) ${option_d}
    E) ${option_e}
    Correct Answer: ${correct_answer_letter.toUpperCase()}
  
    First, think about your task step by step inside <thinking></thinking> tags. Your thinking process should try to go over all the posible ways to solve the question and try to find the most atomic way to solve it.
  
    Then, when you are ready to answer, after your </thinking> tag you should write your response using "Step" as the key and the step as the value. For example:
  
    <thinking>
      Thinking process... Think as long as you need to.
    </thinking>
  
    Step 1: ...
    Step 2: ...
    Step 3: ...
    ...
  
    Important instructions:
    - Don't skip any step.
    - Don't assume anything about the reader, try to solve it in the most atomic and pedagogical way possible.
    - Use as many steps as you need to.
    `
  
    return geminiFlash(prompt)
  }
    
  /**
   * Makes an API call to Gemini Flash model with the provided prompt
  * @customfunction
    * @param {string} prompt The prompt to send to the Gemini model
    * @returns {string} The generated response text from Gemini
    */
  function geminiFlash(prompt) {
    const apiKey = 'AIzaSyBth1qohGO3qZi8hw3tS4ig-vA0c-AZFVk';
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`;
  
    const headers = {
      'Content-Type': 'application/json'
    };
  
    const payload = {
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: prompt
            }
          ]
        }
      ],
      generationConfig: {
        temperature: 1,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 8192,
        responseMimeType: 'text/plain'
      }
    };
  
    const options = {
      method: 'post',
      headers: headers,
      payload: JSON.stringify(payload),
      muteHttpExceptions: true
    };
  
    try {
      const response = UrlFetchApp.fetch(apiUrl, options);
      const data = JSON.parse(response.getContentText());
      // Assuming the response contains generated text in the expected format
      return data.candidates[0].content.parts[0].text.trim();
    } catch (error) {
      Logger.log('Error: ' + error);
      return 'An error occurred: ' + error.message;
    }
  }
  
  /**
   * Makes an API call to Gemini Flash model with the provided prompt
  * @customfunction
    * @param {string} prompt The prompt to send to the Gemini model
    * @returns {string} The generated response text from Gemini
    */
  function geminiFlashLite(prompt) {
    const apiKey = 'AIzaSyBth1qohGO3qZi8hw3tS4ig-vA0c-AZFVk';
    const apiUrl = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key=${apiKey}`;
  
    const headers = {
      'Content-Type': 'application/json'
    };
  
    const payload = {
      contents: [
        {
          role: 'user',
          parts: [
            {
              text: prompt
            }
          ]
        }
      ],
      generationConfig: {
        temperature: 1,
        topK: 40,
        topP: 0.95,
        maxOutputTokens: 8192,
        responseMimeType: 'text/plain'
      }
    };
  
    const options = {
      method: 'post',
      headers: headers,
      payload: JSON.stringify(payload),
      muteHttpExceptions: true
    };
  
    try {
      const response = UrlFetchApp.fetch(apiUrl, options);
      const data = JSON.parse(response.getContentText());
      // Assuming the response contains generated text in the expected format
      return data.candidates[0].content.parts[0].text.trim();
    } catch (error) {
      Logger.log('Error: ' + error);
      return 'An error occurred: ' + error.message;
    }
  }

/**
 * Extracts skills from a question, choices, and step-by-step solution
 * @customfunction
 * @param {string} question The full text of the question
 * @param {string} option_a Text for option A
 * @param {string} option_b Text for option B
 * @param {string} option_c Text for option C
 * @param {string} option_d Text for option D
 * @param {string} option_e Text for option E
 * @param {string} correct_answer_letter The letter of the correct answer (A-E)
 * @param {string} solution The step-by-step solution
 * @returns {string} A numeric array of skills identified in the format [1, 2, 3, 4,...]
 */
function extractSkills(question, option_a, option_b, option_c, option_d, option_e, correct_answer_letter, solution) {
  // Hardcoded skills table in markdown format
  const skillsTable = `
| id | skill                           | category                   | explanation                                                               |
| -- | ------------------------------- | -------------------------- | ------------------------------------------------------------------------- |
| 1  | Addition                        | Arithmetic/Number Sense    | Combining quantities to find their sum                                    |
| 2  | Subtraction                     | Arithmetic/Number Sense    | Finding the difference between quantities                                 |
| 3  | Multiplication                  | Arithmetic/Number Sense    | Repeated addition; finding the product of numbers                         |
| 4  | Division                        | Arithmetic/Number Sense    | Distributing a quantity into equal parts                                  |
| 5  | Mental Math                     | Arithmetic/Number Sense    | Performing calculations quickly without writing or tools                  |
| 6  | Estimation                      | Arithmetic/Number Sense    | Finding approximate values through rounding and reasoning                 |
| 7  | Number Properties               | Arithmetic/Number Sense    | Understanding even/odd prime/composite divisibility rules                 |
| 8  | Fractions                       | Arithmetic/Number Sense    | Working with parts of a whole; operations and simplification              |
| 9  | Decimals                        | Arithmetic/Number Sense    | Working with base-10 representations; operations and conversions          |
| 10 | Percentages                     | Arithmetic/Number Sense    | Representing parts per hundred; percentage change calculations            |
| 11 | Factors and Multiples           | Arithmetic/Number Sense    | Finding common factors prime factorization LCM and GCD                    |
| 12 | Order of Operations             | Arithmetic/Number Sense    | Following PEMDAS/BODMAS conventions for calculation sequence              |
| 13 | Ratios                          | Ratios Proportions Rates   | Comparing quantities using multiplicative relationships                   |
| 14 | Proportions                     | Ratios Proportions Rates   | Setting up and solving equivalent ratios                                  |
| 15 | Rates                           | Ratios Proportions Rates   | Quantities that compare different units (speed density etc.)              |
| 16 | Unit Conversions                | Ratios Proportions Rates   | Converting between different measurement systems                          |
| 17 | Scale Factors                   | Ratios Proportions Rates   | Understanding and applying proportional relationships in models           |
| 18 | Variables and Expressions       | Algebra                    | Using symbols to represent unknown values; simplifying expressions        |
| 19 | Linear Equations                | Algebra                    | Solving first-degree equations; understanding slope-intercept form        |
| 20 | Systems of Equations            | Algebra                    | Solving multiple equations with multiple variables simultaneously         |
| 21 | Inequalities                    | Algebra                    | Solving and graphing relationships with inequality symbols                |
| 22 | Polynomials                     | Algebra                    | Working with expressions containing variables with whole-number exponents |
| 23 | Factoring                       | Algebra                    | Breaking down expressions into products of simpler expressions            |
| 24 | Quadratic Equations             | Algebra                    | Solving second-degree equations using various methods                     |
| 25 | Exponents and Radicals          | Algebra                    | Understanding powers and roots; applying laws of exponents                |
| 26 | Rational Expressions            | Algebra                    | Working with fractions containing variables                               |
| 27 | Logarithms                      | Algebra                    | Understanding and applying log properties and conversions                 |
| 28 | Function Notation               | Functions and Graphing     | Using f(x) notation; evaluating and interpreting functions                |
| 29 | Domain and Range                | Functions and Graphing     | Identifying valid inputs and resulting outputs of functions               |
| 30 | Linear Functions                | Functions and Graphing     | Graphing and interpreting straight lines; understanding slope             |
| 31 | Quadratic Functions             | Functions and Graphing     | Graphing and interpreting parabolas; vertex form                          |
| 32 | Polynomial Functions            | Functions and Graphing     | Graphing and analyzing higher-degree polynomial functions                 |
| 33 | Exponential Functions           | Functions and Graphing     | Understanding and graphing growth and decay models                        |
| 34 | Logarithmic Functions           | Functions and Graphing     | Graphing and applying log functions as inverses of exponentials           |
| 35 | Piecewise Functions             | Functions and Graphing     | Analyzing functions defined differently over different domains            |
| 36 | Function Transformations        | Functions and Graphing     | Applying shifts stretches compressions and reflections                    |
| 37 | Function Composition            | Functions and Graphing     | Creating new functions by applying one function to another                |
| 38 | Angles                          | Geometry                   | Measuring and classifying angles; angle relationships                     |
| 39 | Triangles                       | Geometry                   | Properties of different triangle types; sum of angles                     |
| 40 | Quadrilaterals                  | Geometry                   | Properties of squares rectangles parallelograms trapezoids                |
| 41 | Polygons                        | Geometry                   | Properties of regular and irregular polygons                              |
| 42 | Circles                         | Geometry                   | Properties involving radius diameter chord arc sector                     |
| 43 | Area and Perimeter              | Geometry                   | Calculating measurements for 2D shapes                                    |
| 44 | Volume and Surface Area         | Geometry                   | Calculating measurements for 3D shapes                                    |
| 45 | Coordinate Geometry             | Geometry                   | Working with shapes on the coordinate plane; distance formula             |
| 46 | Transformations                 | Geometry                   | Applying translations rotations reflections dilations                     |
| 47 | Pythagorean Theorem             | Geometry                   | Understanding and applying a^2 + b^2 = c^2 in right triangles             |
| 48 | Similar Triangles               | Geometry                   | Working with proportional relationships in triangles                      |
| 49 | Congruence                      | Geometry                   | Identifying and proving shapes are identical                              |
| 50 | Geometric Constructions         | Geometry                   | Creating precise geometric figures using compass and straightedge         |
| 51 | Geometric Proofs                | Geometry                   | Developing logical arguments to verify geometric properties               |
| 52 | Basic Trigonometric Ratios      | Trigonometry               | Understanding sine cosine tangent in right triangles                      |
| 53 | Inverse Trigonometric Functions | Trigonometry               | Using arcsin arccos arctan to find angles                                 |
| 54 | Unit Circle                     | Trigonometry               | Understanding trigonometric values on the unit circle                     |
| 55 | Trigonometric Identities        | Trigonometry               | Applying fundamental relationships between trig functions                 |
| 56 | Law of Sines/Cosines            | Trigonometry               | Solving non-right triangles using advanced trigonometric laws             |
| 57 | Polar Coordinates               | Trigonometry               | Representing points using distance and angle from origin                  |
| 58 | Mean Median Mode                | Statistics and Probability | Calculating and interpreting measures of central tendency                 |
| 59 | Range and Standard Deviation    | Statistics and Probability | Calculating and interpreting measures of dispersion                       |
| 60 | Data Representation             | Statistics and Probability | Creating and interpreting graphs charts and histograms                    |
| 61 | Basic Probability               | Statistics and Probability | Calculating chances of events occurring                                   |
| 62 | Compound Probability            | Statistics and Probability | Working with AND/OR probabilities and independence                        |
| 63 | Conditional Probability         | Statistics and Probability | Finding probability of events given other events occurred                 |
| 64 | Combinatorics                   | Statistics and Probability | Counting principles permutations and combinations                         |
| 65 | Statistical Inference           | Statistics and Probability | Making predictions based on sample data                                   |
| 66 | Regression Analysis             | Statistics and Probability | Finding relationships between variables in data sets                      |
| 67 | Hypothesis Testing              | Statistics and Probability | Evaluating statistical claims using data                                  |
| 68 | Limits                          | Calculus                   | Understanding behavior of functions as inputs approach specific values    |
| 69 | Derivatives                     | Calculus                   | Finding rates of change; differential calculus                            |
| 70 | Integrals                       | Calculus                   | Finding areas and accumulation; integral calculus                         |
| 71 | Applications of Calculus        | Calculus                   | Optimization problems related rates area under curves                     |
| 72 | Word Problem Translation        | Problem Solving            | Converting text descriptions into mathematical expressions                |
| 73 | Multi-step Problem Solving      | Problem Solving            | Breaking complex problems into manageable steps                           |
| 74 | Mathematical Modeling           | Problem Solving            | Creating math representations of real-world scenarios                     |
| 75 | Logical Reasoning               | Problem Solving            | Developing valid arguments and identifying fallacies                      |
| 76 | Pattern Recognition             | Problem Solving            | Identifying and extending numerical and geometric patterns                |
| 77 | Complex Numbers                 | Advanced Topics            | Working with numbers involving imaginary units                            |
| 78 | Matrices                        | Advanced Topics            | Operations with arrays of numbers; linear transformations                 |
| 79 | Sequences and Series            | Advanced Topics            | Working with ordered lists of numbers and their sums                      |
| 80 | Vector Geometry                 | Advanced Topics            | Understanding directed quantities; dot and cross products                 |
| 81 | Set Theory                      | Discrete Mathematics       | Working with collections of objects; set operations                       |
| 82 | Graph Theory                    | Discrete Mathematics       | Analyzing networks of nodes and connections                               |
| 83 | Financial Mathematics           | Applied Mathematics        | Interest calculations budgeting and financial planning                    |
| 84 | Technology in Mathematics       | Tools and Technology       | Using calculators spreadsheets and software effectively                   |
| 85 | Mathematical Communication      | Meta Skills                | Explaining mathematical thinking clearly and precisely                    |
| 86 | Mathematical Literacy           | Meta Skills                | Reading and interpreting mathematical notation and text                   |
`;

  const prompt = `
You are a skilled math education analyst. Your task is to identify which mathematical skills are demonstrated in the following question and its solution.

Question: ${question}
A) ${option_a}
B) ${option_b}
C) ${option_c}
D) ${option_d}
E) ${option_e}
Correct Answer: ${correct_answer_letter.toUpperCase()}
Step-by-step solution: ${solution}

Here is a list of skills you should consider:
### Begin Skills Table ###
${skillsTable}
### End Skills Table ###

For you answer first you have to think step by step about the question and solution inside <thinking></thinking> tags. Carefully analyze the question and solution to identify which skills are being tested or applied. Consider the mathematical concepts involved, the steps required to solve the problem, and the specific techniques used in the solution.

For each skill in the table, evaluate whether it's significantly used or tested in this problem. Be selective and only choose skills that are directly relevant to solving this specific problem.

After your </thinking> tag, you should write your response as a JSON object with a "skills" key. The value should be an array of skill IDs as integers. 

For example, your response should look like this:

<thinking>
  Thinking process... Think as long as you need to.
</thinking>

{
  "skills": [1, 2, 3, ...]
}

Only include skills that are clearly demonstrated in the question or required for its solution.
`;

  try {
    // Run 5 separate LLM calls
    const allSkillsArrays = [];
    
    for (let i = 0; i < 5; i++) {
      const response = geminiFlash(prompt);
      // Extract the JSON from the response
      const jsonMatch = response.match(/\{\s*"skills"\s*:\s*\[.*?\]\s*\}/s);
      
      if (jsonMatch) {
        const skillsJson = JSON.parse(jsonMatch[0]);
        allSkillsArrays.push(skillsJson.skills);
      } else {
        // If no JSON was found, attempt to extract just the array
        const arrayMatch = response.match(/\[\s*\d+(?:\s*,\s*\d+)*\s*\]/);
        if (arrayMatch) {
          allSkillsArrays.push(JSON.parse(arrayMatch[0]));
        } else {
          // If we can't extract skills, push an empty array
          allSkillsArrays.push([]);
        }
      }
    }
    
    // Count occurrences of each skill across all responses
    const skillCounts = {};
    for (const skillsArray of allSkillsArrays) {
      for (const skill of skillsArray) {
        skillCounts[skill] = (skillCounts[skill] || 0) + 1;
      }
    }
    
    // Keep only skills that appear in at least 2 of the 3 responses
    const consensusSkills = Object.keys(skillCounts)
      .filter(skill => skillCounts[skill] >= 3)
      .map(skill => parseInt(skill, 10))
      .sort((a, b) => a - b);
    
    return JSON.stringify(consensusSkills);
  } catch (error) {
    Logger.log('Error: ' + error);
    return 'An error occurred: ' + error.message;
  }
}

/**
 * Extracts the cognitive level of a question based on Bloom's taxonomy.
 * @customfunction
 * @param {string} question The question text
 * @param {string} option_a Option A text
 * @param {string} option_b Option B text
 * @param {string} option_c Option C text
 * @param {string} option_d Option D text
 * @param {string} option_e Option E text
 * @param {string} correct_answer_letter The letter of the correct answer
 * @param {string} solution The step-by-step solution
 * @returns {string} The cognitive level (1-6) based on Bloom's taxonomy
 */
function extractCognitiveLevel(question, option_a, option_b, option_c, option_d, option_e, correct_answer_letter, solution) {
  // Bloom's taxonomy cognitive levels in a detailed rubric format
  const bloomsRubric = `
| level | name          | definition                                                                        | indicators                                                                                                        | keywords                                    |
| ----- | ------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| 1     | Remembering   | Students recall or recognize basic facts, formulas, or definitions                | Questions that ask for direct recall of a formula or definition (e.g., Which formula is correct?)                 | define, recall, identify, list, recognize   |
| 2     | Understanding | Students restate or interpret concepts in their own words                         | Questions that ask for the best interpretation or explanation (e.g., Which option best explains this concept?)    | explain, interpret, describe, summarize     |
| 3     | Applying      | Students use knowledge or concepts to solve routine problems in familiar contexts | Straightforward procedure or calculation (e.g., Which answer is the correct solution?)                            | apply, solve, calculate, use, illustrate    |
| 4     | Analyzing     | Students break down relationships, compare methods, or find underlying patterns   | Questions may show different solution paths, requiring identification of the correct reasoning or spotting errors | analyze, compare, contrast, differentiate   |
| 5     | Evaluating    | Students judge or critique ideas based on evidence or criteria                    | Often presents a worked solution to critique (e.g., Which statement best justifies or identifies the error?)      | evaluate, critique, judge, justify          |
| 6     | Creating      | Students produce or design something novel or adapt knowledge to new contexts     | Tricky in multiple choice; may involve selecting the best new or synthesized approach from multiple proposals     | create, design, develop, propose, formulate |
`;

  const prompt = `
You are an expert in educational assessment and Bloom's taxonomy. Your task is to determine the cognitive level of the following math question according to Bloom's taxonomy.

Here is the question:

### Begin Question ###
Question: ${question}
A) ${option_a}
B) ${option_b}
C) ${option_c}
D) ${option_d}
E) ${option_e}
Correct Answer: ${correct_answer_letter.toUpperCase()}

Step-by-step solution: ${solution}
### End Question ###

Here is the Bloom's taxonomy cognitive levels rubric:

### Begin Bloom's Taxonomy Rubric ###
${bloomsRubric}
### End Bloom's Taxonomy Rubric ###

For your answer, first think step by step about the question and solution inside <thinking></thinking> tags. Carefully analyze:
1. What cognitive skills are required to solve this problem?
2. What mental processes must the student engage in?
3. Is the student mainly recalling information, understanding concepts, applying knowledge, analyzing relationships, evaluating solutions, or creating something new?
4. Which level in Bloom's taxonomy best matches the highest cognitive demand of this question?

After your </thinking> tag, provide your response as a JSON object with a "level" key. The value should be an integer from 1 to 6 representing the cognitive level according to Bloom's taxonomy.

For example, your response should look like this:

<thinking>
  Detailed thinking process about the cognitive demands of the question...
</thinking>

{
  "level": 3
}

Be sure to only return a single integer from 1 to 6 representing the highest cognitive level required to solve the problem.
`;

  try {
    // Run 3 separate LLM calls to get a consensus
    const allLevels = [];
    
    for (let i = 0; i < 3; i++) {
      const response = geminiFlash(prompt);
      // Extract the JSON from the response
      const jsonMatch = response.match(/\{\s*"level"\s*:\s*(\d+)\s*\}/);
      
      if (jsonMatch && jsonMatch[1]) {
        const level = parseInt(jsonMatch[1], 10);
        if (level >= 1 && level <= 6) {  // Validate the level is within Bloom's taxonomy range
          allLevels.push(level);
        }
      } else {
        // If no JSON was found, attempt to extract just the number
        const levelMatch = response.match(/level"?\s*:?\s*(\d+)/i);
        if (levelMatch && levelMatch[1]) {
          const level = parseInt(levelMatch[1], 10);
          if (level >= 1 && level <= 6) {
            allLevels.push(level);
          }
        }
      }
    }
    
    // Calculate the mode (most common value) as the consensus
    if (allLevels.length === 0) {
      return "Unable to determine cognitive level";
    }
    
    const levelCounts = {};
    let maxCount = 0;
    let consensusLevel = null;
    
    for (const level of allLevels) {
      levelCounts[level] = (levelCounts[level] || 0) + 1;
      if (levelCounts[level] > maxCount) {
        maxCount = levelCounts[level];
        consensusLevel = level;
      }
    }
    
    return consensusLevel.toString();
  } catch (error) {
    Logger.log('Error: ' + error);
    return 'An error occurred: ' + error.message;
  }
}

/**
 * Identifies and lists common misconceptions students might have when solving a math problem.
 * Returns three values: the thinking process, array of misconceptions, and count of misconceptions.
 * @param {string} question The question text
 * @param {string} option_a Option A text
 * @param {string} option_b Option B text
 * @param {string} option_c Option C text
 * @param {string} option_d Option D text
 * @param {string} option_e Option E text
 * @param {string} correct_answer_letter The letter of the correct answer
 * @param {string} solution The step-by-step solution
 * @returns {Array} A row array containing [thinking process, misconceptions array, count]
 * @customfunction
 */
function listMisconceptions(question, option_a, option_b, option_c, option_d, option_e, correct_answer_letter, solution) {
  const prompt = `
You are an expert in math education who specializes in identifying common student misconceptions. Your task is to analyze the following math question and identify all potential misconceptions students might have when solving it.

### Begin Question ###
Question: ${question}
A) ${option_a}
B) ${option_b}
C) ${option_c}
D) ${option_d}
E) ${option_e}
Correct Answer: ${correct_answer_letter.toUpperCase()}

Step-by-step solution: ${solution}
### End Question ###

First, think step by step about the question and solution inside <thinking></thinking> tags. Carefully analyze:
1. What are the key concepts and operations involved in this problem?
2. What are the common errors students make with these concepts?
3. What might students misunderstand about the problem statement itself?
4. Which incorrect answer choices might attract students, and why?
5. What procedural errors might occur during the solution process?
6. Are there any known conceptual frameworks or mental models that students commonly apply incorrectly to this type of problem?

Your list of misconceptions should be MECE (Mutually Exclusive, Collectively Exhaustive):
- Mutually Exclusive: Each misconception should be distinct and not overlap significantly with others
- Collectively Exhaustive: The list should cover all reasonable potential misconceptions students might have

After your </thinking> tag, provide your response as a JSON object with a "misconceptions" key. The value should be an array of strings, each describing a specific misconception. Be detailed and specific about each misconception.

For example, your response might look like this:

<thinking>
  Detailed thinking process about potential misconceptions...
</thinking>

{
  "misconceptions": [
    "Students might apply the wrong formula by confusing concept X with concept Y, leading them to calculate Z incorrectly",
    "Students might misinterpret the meaning of term A in the question, causing them to solve for the wrong variable",
    "Students might make a sign error when applying operation B, resulting in the opposite of the correct answer",
    "Students might forget to convert units properly, leading to an answer that's off by a factor of C"
  ]
}

Be comprehensive and list all reasonable misconceptions a student might have with this specific problem. Each misconception should be clearly explained.
`;

  try {
    // Make a single LLM call to get misconceptions
    const response = geminiFlash(prompt);
    let misconceptionsArray = [];
    let thinkingProcess = "";
    
    // Extract the thinking process
    const thinkingMatch = response.match(/<thinking>([\s\S]*?)<\/thinking>/);
    if (thinkingMatch && thinkingMatch[1]) {
      thinkingProcess = thinkingMatch[1].trim();
    }
    
    // Extract the JSON from the response
    const jsonMatch = response.match(/\{\s*"misconceptions"\s*:\s*\[([\s\S]*?)\]\s*\}/);
    
    if (jsonMatch && jsonMatch[1]) {
      // Parse the array of misconceptions from the JSON
      const misconceptionsText = jsonMatch[1];
      // Use regex to split by commas that aren't inside quotes
      const misconceptionParts = misconceptionsText.split(/,(?=\s*")/);
      
      for (let part of misconceptionParts) {
        // Clean up each misconception string
        const misconception = part.trim().replace(/^"|"$/g, '').trim();
        if (misconception) {
          misconceptionsArray.push(misconception);
        }
      }
    } else {
      // Fallback: Try to find an array-like structure
      const arrayMatch = response.match(/\[\s*"([\s\S]*?)"\s*\]/);
      if (arrayMatch && arrayMatch[1]) {
        const misconceptionsText = arrayMatch[1];
        const misconceptionParts = misconceptionsText.split(/",\s*"/);
        
        for (let part of misconceptionParts) {
          const misconception = part.trim().replace(/^"|"$/g, '').trim();
          if (misconception) {
            misconceptionsArray.push(misconception);
          }
        }
      }
    }
    
    // Get the count of misconceptions
    const misconceptionsCount = misconceptionsArray.length;
    
    // Create a special data structure for Google Sheets
    // A 2D array with a single row will spill horizontally
    const result = [[
      thinkingProcess, 
      JSON.stringify(misconceptionsArray), 
      misconceptionsCount
    ]];
    
    // If we couldn't extract any misconceptions, return empty values
    if (misconceptionsCount === 0) {
      return [[
        "No thinking process extracted.", 
        "[]", 
        0
      ]];
    }
    
    // Return a 2D array with a single row to ensure horizontal spillage
    return result;
  } catch (error) {
    Logger.log('Error: ' + error);
    return [[
      "Error occurred during processing.", 
      "[]", 
      0
    ]];
  }
}