#!/usr/bin/env python3
import pymongo


def top_students(mongo_collection):
    """
    docs go here!
    """
    students_with_avg = []
    
    students_cursor = mongo_collection.find({})
    
    for student in students_cursor:
        total_score = 0
        num_topics = 0
        
        if 'topics' in student and isinstance(student['topics'], list):
            for topic in student['topics']:
                if isinstance(topic, dict) and 'score' in topic and isinstance(topic['score'], (int, float)):
                    total_score += topic['score']
                    num_topics += 1
        
        if num_topics > 0:
            average_score = total_score / num_topics
        else:
            average_score = 0
            
        student_copy = student.copy()
        student_copy['averageScore'] = average_score
        students_with_avg.append(student_copy)
            
    students_with_avg.sort(key=lambda s: s['averageScore'], reverse=True)
    
    return students_with_avg
