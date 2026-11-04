/**
 * Pure field-intent matcher (no imports, fully unit-testable).
 * Fixes the P0 routing bugs found in review:
 *  - father/mother name queries had NO synonym group at all
 *  - first-match substring order let bare 'total' steal "total marks" queries
 *    from the total_marks group (marksheet misroute)
 *  - matching is now score-based: longer (more specific) keyword wins,
 *    so "father name" beats the generic 'name' in candidate_name.
 */

export interface SynonymGroup {
  canonical: string;
  label: string;
  keywords: string[];
}

export const FIELD_SYNONYMS: SynonymGroup[] = [
  {
    canonical: 'roll_number',
    label: 'Roll Number',
    keywords: ['roll number', 'roll no', 'enrollment number', 'enrolment number', 'enrollment', 'enrolment', 'roll', 'anukramank']
  },
  {
    canonical: 'candidate_name',
    label: 'Student / Candidate Name',
    keywords: ['candidate name', 'student name', 'mera naam', 'account holder', 'insured name', 'naam', 'name']
  },
  {
    canonical: 'father_name',
    label: "Father's Name",
    keywords: ['father name', "father's name", 'pita ka naam', 'papa ka naam', 'father', 'son of', 's/o']
  },
  {
    canonical: 'mother_name',
    label: "Mother's Name",
    keywords: ['mother name', "mother's name", 'mata ka naam', 'maa ka naam', 'mother', 'daughter of', 'd/o']
  },
  {
    canonical: 'parent_name',
    label: 'Parent / Guardian Name',
    keywords: ['parent name', 'parents name', 'guardian name', 'guardian']
  },
  {
    canonical: 'date_of_birth',
    label: 'Date of Birth (DOB)',
    keywords: ['date of birth', 'dob', 'birth date', 'janam tithi', 'janam din', 'birthday', 'born']
  },
  {
    canonical: 'due_date',
    label: 'Due Date',
    keywords: ['due date', 'last date', 'aakhri tarikh', 'kab tak bharna hai', 'payment due', 'bill due date']
  },
  {
    canonical: 'total_amount',
    label: 'Bill / Payable Amount',
    // NOTE: bare 'total' removed — it stole "total marks" queries (P0 misroute)
    keywords: ['bill amount', 'amount due', 'total amount', 'net payable', 'kitna bill', 'bill kitna', 'kitne ka hai', 'kitna paisa']
  },
  {
    canonical: 'account_number',
    label: 'Bank Account Number',
    keywords: ['account number', 'a/c no', 'account no', 'khata sankhya', 'bank account', 'khata number']
  },
  {
    canonical: 'ifsc_code',
    label: 'IFSC Code',
    keywords: ['ifsc', 'ifsc code', 'branch code', 'bank code']
  },
  {
    canonical: 'available_balance',
    label: 'Account Balance',
    keywords: ['available balance', 'kitne paise hai', 'kitna balance', 'bank balance', 'balance']
  },
  {
    canonical: 'consumer_number',
    label: 'Consumer Number (CA)',
    keywords: ['consumer number', 'consumer no', 'ca number', 'ca no', 'upbhokta sankhya', 'connection number']
  },
  {
    canonical: 'policy_number',
    label: 'Insurance Policy Number',
    keywords: ['policy number', 'policy no', 'bima number', 'insurance number', 'policy id']
  },
  {
    canonical: 'sum_insured',
    label: 'Sum Insured Coverage',
    keywords: ['sum insured', 'coverage amount', 'bima rashi', 'kitna insurance', 'coverage']
  },
  {
    canonical: 'cgpa',
    label: 'Cumulative CGPA / Grade',
    keywords: ['cgpa', 'sgpa', 'gpa', 'grade']
  },
  {
    canonical: 'total_marks',
    label: 'Total Marks',
    keywords: ['total marks', 'marks', 'kitne marks', 'kitna number', 'number kitne', 'marks kitne', 'grand total', 'score']
  },
  {
    canonical: 'percentage',
    label: 'Percentage',
    keywords: ['percentage', 'percent', 'pratishat', 'kitna percent', 'division', 'result']
  },
  {
    canonical: 'board_name',
    label: 'Examination Board',
    keywords: ['which board', 'kaun sa board', 'examination board', 'board name', 'board', 'university']
  },
  {
    canonical: 'school_name',
    label: 'School / Institution',
    keywords: ['school name', 'college name', 'school ka naam', 'college', 'school', 'institution']
  },
  {
    canonical: 'year_of_passing',
    label: 'Passing Year',
    // bare 'year' removed — too broad
    // 'session' moved to its own group: a session range (2024-2028) is not a
    // passing year, and the duplicate keyword made matching order-dependent.
    keywords: ['passing year', 'kab pass kiya', 'kon se saal', 'year of passing']
  },
  {
    canonical: 'aadhaar_number',
    label: 'Aadhaar Card Number',
    keywords: ['aadhaar number', 'aadhaar no', 'uidai number', 'aadhar number']
  },
  {
    canonical: 'pan_number',
    label: 'PAN Card Number',
    keywords: ['pan number', 'pan no', 'pan card number']
  },
  {
    canonical: 'residential_address',
    label: 'Residential Address',
    keywords: ['residential address', 'ghar ka pata', 'service address', 'address', 'pata']
  },
  {
    canonical: 'blood_group',
    label: 'Blood Group',
    // 'group' alone is far too broad — always require the blood context
    keywords: ['blood group', 'blood type', 'blood grp', 'rakt samuh', 'rakt group']
  },
  {
    canonical: 'contact_number',
    label: 'Contact Number',
    keywords: ['contact number', 'contact no', 'mobile number', 'mobile no', 'phone number', 'phone no', 'mobile']
  },
  {
    canonical: 'student_id',
    label: 'Student ID',
    keywords: ['student id', 'enrollment id', 'enrolment id', 'student code']
  },
  {
    canonical: 'course_branch',
    label: 'Course / Branch',
    keywords: ['course and branch', 'course/branch', 'course branch', 'which course', 'kaun sa course', 'branch', 'course']
  },
  {
    canonical: 'session',
    label: 'Session / Validity',
    keywords: ['session', 'validity', 'valid till', 'valid up to', 'kab tak valid']
  }
];

export interface FieldMatch {
  group: SynonymGroup;
  score: number;
}

function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Score-based group selection: every distinct keyword that matches adds its
 * length to the group's score (longer keyword = more specific signal).
 * Returns the highest-scoring group, or null when nothing matches.
 */
export function matchFieldGroup(lowerQuery: string): FieldMatch | null {
  const query = lowerQuery.toLowerCase();
  let best: FieldMatch | null = null;

  for (const group of FIELD_SYNONYMS) {
    let score = 0;
    for (const keyword of group.keywords) {
      const escaped = escapeRegExp(keyword);
      const wordBoundaryHit = new RegExp(`\\b${escaped}\\b`, 'i').test(query);
      const includesHit = query.includes(keyword);
      if (wordBoundaryHit || includesHit) {
        // Word-boundary hits count slightly higher than loose substring hits
        score += keyword.length + (wordBoundaryHit ? 2 : 0);
      }
    }
    if (score > 0 && (!best || score > best.score)) {
      best = { group, score };
    }
  }

  return best;
}

/** Canonical fields that answer a "parent / guardian" question. */
export const PARENT_CANDIDATE_FIELDS = ['father_name', 'mother_name', 'guardian_name'];
