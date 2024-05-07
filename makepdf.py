import pdfkit
import markdown

text = '''
**Storyline:**

Imagine you're at a theme park, and your friend, Alex, just won a pair of 
roller coaster tickets in a game. The roller coaster is called "Projectile
Pandemonium," and it's an intense ride that simulates the motion of 
projectiles. As you wait in line, Alex explains the physics behind the 
ride to you.

**Text:**

---

### Introduction

Projectiles are objects that move through the air or space under the 
influence of gravity. We've all seen projectiles in action - from a 
baseball soaring through the air to a rocket blasting off into space. In 
this lesson, we'll explore the basics of projectile motion and how it 
applies to our daily lives.

### What is Projectile Motion?

Projectile motion is the study of objects that move under the influence of
gravity. It's a fundamental concept in physics that helps us understand 
many natural phenomena, from the flight of birds to the trajectory of 
bullets.

### The Five Key Factors

To fully grasp projectile motion, we need to consider five key factors:

1. **Initial Velocity**: The speed at which an object starts moving.
2. **Angle of Projection**: The angle at which an object is thrown or 
launched.
3. **Gravity**: The force that pulls objects towards the ground.
4. **Time**: How long it takes for the projectile to reach its maximum 
height.
5. **Range**: The distance traveled by the projectile from launch to 
impact.

### Equations of Motion

Now that we've covered the basics, let's dive into some equations! These 
will help us predict where projectiles land and how high they fly.

* **Initial Velocity (v0)**: v0 = √(2\*h/g) where h is the initial height 
and g is the acceleration due to gravity.
* **Range (R)**: R = (v0^2 \* sin(2θ)) / g where θ is the angle of 
projection.
* **Maximum Height (h_max)**: h_max = (v0^2 \* sin^2(θ)) / (2\*g)

### Quizzes and Problems

Now it's your turn to practice what you've learned! Try solving these 
quizzes and problems:

1. A baseball is thrown at an angle of 45° with an initial velocity of 90 
km/h. How far does it travel?
	* Answer: R = (90^2 \* sin(90)) / 9.8 ≈ 123.5 meters
2. A bullet is fired from a gun at an angle of 30° with an initial 
velocity of 250 m/s. What's its maximum height?
	* Answer: h_max = (250^2 \* sin^2(30)) / (2\*9.8) ≈ 43.5 meters

### Conclusion

And that's it! You now have a solid understanding of projectile motion and
how to apply it to real-world scenarios. Remember, the next time you watch
a rocket soar into space or a baseball sail through the air, you can 
appreciate the physics behind it.

'''

# Convert Markdown to HTML
html_text = markdown.markdown(text)

# Create a PDF file

pdfkit.from_string(html_text, 'projectile_motion.pdf')
    

print("PDF created successfully!")
