from manim import *
# from manim_voice import SpeechManager, SpeechServer

class MagicalPyramid(Scene):
    def construct(self):
        # Set up speech manager
        # speech_manager = SpeechManager()

        # Introduction
        intro = Text("The Magical Pyramid of Mathoria", font_size=48)
        self.play(Write(intro))
        # speech_manager.say_text("The Magical Pyramid of Mathoria")
        self.wait(2)
        self.play(FadeOut(intro))

        # Characters
        # timmy = ImageMobject("timmy.png").scale(0.5)
        # emma = ImageMobject("emma.png").scale(0.5)
        # timmy.shift(LEFT * 2)
        # emma.shift(RIGHT * 2)
        # self.play(FadeIn(timmy, emma))
        # speech_manager.say_text("Once upon a time, in a magical kingdom called 'Mathoria,' there lived two best friends, Timmy and Emma.")
        self.wait(2)

        # Ancient Scroll
        # scroll = ImageMobject("scroll.png").scale(1.5)
        # self.play(FadeIn(scroll))
        # speech_manager.say_text("One sunny day, they stumbled upon an ancient scroll hidden deep within the forest.")
        self.wait(2)

        # Triangle
        triangle = Polygon(ORIGIN, RIGHT * 3, RIGHT * 3 + UP * 4)
        a = Tex("A = 3").next_to(triangle.get_side(UP + RIGHT), UP)
        b = Tex("B = 4").next_to(triangle.get_side(RIGHT), RIGHT)
        c = Tex("C = ?").next_to(triangle.get_side(LEFT + UP), LEFT)
        self.play(DrawBorderThenFill(triangle), Write(a), Write(b), Write(c))
        # speech_manager.say_text("The scroll revealed a mysterious triangle with three sides: A, B, and C. Side A was 3 inches long, side B was 4 inches long, and side C was hidden.")
        self.wait(2)

        # Pythagorean Theorem
        theorem = MathTex(r"$A^2 + B^2 = C^2$")
        theorem.shift(UP * 2)
        self.play(Write(theorem))
        # speech_manager.say_text("They discovered something amazing! When they added the squares of sides A and B together, it equaled the square of side C.")
        self.wait(2)

        # Testing the Theorem
        new_triangle = Polygon(ORIGIN, RIGHT * 5, RIGHT * 5 + UP * 3)
        new_a = MathTex("A = 5").next_to(new_triangle.get_side(UP + RIGHT), UP)
        new_b = MathTex("B = 3").next_to(new_triangle.get_side(RIGHT), RIGHT)
        new_c = MathTex("C = ?").next_to(new_triangle.get_side(LEFT + UP), LEFT)
        self.play(FadeOut(triangle, a, b, c))
        self.play(DrawBorderThenFill(new_triangle), Write(new_a), Write(new_b), Write(new_c))
        # speech_manager.say_text("They drew a new triangle with different sides and repeated the process. To their delight, the result was always the same: A squared plus B squared equals C squared.")
        self.wait(2)

        # Conclusion
        conclusion = Text("And so, Timmy and Emma's adventure in Mathoria came to an end.", font_size=36)
        # self.play(FadeOut(new_triangle, new_a, new_b, new_c, theorem, timmy, emma, scroll))
        self.play(Write(conclusion))
        # speech_manager.say_text("And so, Timmy and Emma's adventure in Mathoria came to an end. They left the forest with a newfound appreciation for the Pythagorean Theorem and its many applications.")
        self.wait(4)

        # Clean up speech manager
        # speech_manager.terminate()